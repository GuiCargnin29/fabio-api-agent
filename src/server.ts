import "dotenv/config";
import Fastify from "fastify";
import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import { z } from "zod";
import { runWorkflow } from "./agent.js";
import { randomUUID } from "node:crypto";
import { OpenAI } from "openai";
import { toFile } from "openai/uploads";

const envSchema = z.object({
  OPENAI_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),
  APP_API_KEY: z.string().min(16, "APP_API_KEY must be at least 16 chars"),
  PORT: z.coerce.number().int().positive().default(3000),
  LOG_LEVEL: z.string().optional(),
  NODE_ENV: z.string().optional()
});

const env = envSchema.parse(process.env);
const openai = new OpenAI({ apiKey: env.OPENAI_API_KEY, timeout: 120000 });

const fastify = Fastify({
  logger: {
    level: env.LOG_LEVEL ?? "info"
  },
  bodyLimit: 25_000_000
});

await fastify.register(cors, {
  origin: true
});

await fastify.register(helmet);

await fastify.register(rateLimit, {
  max: 60,
  timeWindow: "1 minute"
});

fastify.setErrorHandler((error, _request, reply) => {
  const status = (error as any)?.statusCode ?? 500;
  const code = status >= 500 ? "internal_error" : "bad_request";
  const message = status >= 500 ? "Unexpected error" : error.message;

  reply.status(status).send({ error: code, message });
});

const MAX_FILE_BYTES = 10 * 1024 * 1024;
const MAX_ATTACHMENTS_PER_MESSAGE = 7;
const ALLOWED_MIME_TYPES = new Set([
  "application/pdf",
  "image/jpeg",
  "image/png"
]);

const runSchema = z.object({
  chat_id: z.string().min(1, "chat_id is required"),
  input_as_text: z.string().min(1, "input_as_text is required"),
  attachment_ids: z.array(z.string().min(1)).max(MAX_ATTACHMENTS_PER_MESSAGE).optional()
});

const uploadSchema = z.object({
  chat_id: z.string().min(1, "chat_id is required"),
  filename: z.string().min(1, "filename is required"),
  mime_type: z.string().min(1, "mime_type is required"),
  content_base64: z.string().min(1, "content_base64 is required")
});

type RunBody = z.infer<typeof runSchema>;
type UploadBody = z.infer<typeof uploadSchema>;

type AttachmentRecord = {
  attachment_id: string;
  chat_id: string;
  file_id: string;
  filename: string;
  mime_type: string;
  size_bytes: number;
  created_at: string;
};

type WorkflowRunInput = RunBody & {
  attachments?: AttachmentRecord[];
};

const attachmentsById = new Map<string, AttachmentRecord>();
const attachmentsByChatId = new Map<string, string[]>();

function addAttachment(attachment: AttachmentRecord) {
  attachmentsById.set(attachment.attachment_id, attachment);
  const current = attachmentsByChatId.get(attachment.chat_id) ?? [];
  attachmentsByChatId.set(attachment.chat_id, [...current, attachment.attachment_id]);
}

function decodeBase64File(contentBase64: string) {
  const normalized = contentBase64.replace(/^data:[^;]+;base64,/, "");
  const buffer = Buffer.from(normalized, "base64");
  if (!buffer.length) {
    throw new Error("content_base64 is empty or invalid");
  }
  return buffer;
}

function resolveAttachments(chatId: string, attachmentIds: string[] | undefined): AttachmentRecord[] {
  const ids = attachmentIds ?? attachmentsByChatId.get(chatId) ?? [];
  if (ids.length > MAX_ATTACHMENTS_PER_MESSAGE) {
    throw new Error(`Maximum ${MAX_ATTACHMENTS_PER_MESSAGE} attachments per message`);
  }

  const resolved: AttachmentRecord[] = [];
  for (const id of ids) {
    const attachment = attachmentsById.get(id);
    if (!attachment) {
      throw new Error(`Attachment not found: ${id}`);
    }
    if (attachment.chat_id !== chatId) {
      throw new Error(`Attachment does not belong to chat_id: ${id}`);
    }
    resolved.push(attachment);
  }
  return resolved;
}

type DocBlock = {
  block_id?: string;
  type?: string;
  text?: string;
  ordered?: boolean;
  items?: string[];
  rows?: string[][];
  source?: string;
  signatures?: Array<{ name: string; oab: string }>;
};

type DocSection = {
  ordem?: string | number;
  titulo_literal?: string;
  blocks?: DocBlock[];
};

type FinalDoc = {
  doc?: {
    title?: string;
    sections?: DocSection[];
  };
};

function normalizeFinalJson(result: unknown) {
  if (!result || typeof result !== "object") return result;
  const doc = (result as FinalDoc).doc;
  if (!doc || !Array.isArray(doc.sections)) return result;

  const sections = doc.sections;
  let fechoSection: DocSection | undefined;
  let fechoIndex = -1;
  let localDataSection: DocSection | undefined;
  let localDataIndex = -1;
  let signaturesBlock: DocBlock | undefined;

  for (let i = 0; i < sections.length; i += 1) {
    const s = sections[i];
    const blocks = Array.isArray(s.blocks) ? s.blocks : [];
    const hasFechoBlock = blocks.some((b) => b.block_id === "fecho");
    const hasLocalDataBlock = blocks.some((b) => b.block_id === "local_data_assinatura_oab");
    const hasSignatures = blocks.find((b) => b.type === "signatures");
    if (!fechoSection && (hasFechoBlock || (s.titulo_literal ?? "").toLowerCase().includes("termos"))) {
      fechoSection = s;
      fechoIndex = i;
    }
    if (!localDataSection && hasLocalDataBlock) {
      localDataSection = s;
      localDataIndex = i;
    }
    if (!signaturesBlock && hasSignatures) {
      signaturesBlock = hasSignatures;
    }
  }

  const localDataText =
    localDataSection?.blocks?.find((b) => b.block_id === "local_data_assinatura_oab")?.text ??
    "";

  if (!fechoSection) return result;

  const fechoBlocks = Array.isArray(fechoSection.blocks) ? fechoSection.blocks : [];
  const fechoTextBlock = fechoBlocks.find((b) => b.block_id === "fecho");
  const baseFechoLine =
    (fechoTextBlock?.text && fechoTextBlock.text.trim()) ||
    "Termos em que, pede deferimento.";

  const localLine =
    (localDataText && localDataText.trim()) || "Cidade, [PREENCHER: data].";

  const normalizedFechoText = `${baseFechoLine}\n${localLine}`;

  const newFechoBlock: DocBlock = {
    block_id: "fecho",
    type: "paragraph",
    text: normalizedFechoText
  };

  const newSignaturesBlock: DocBlock =
    signaturesBlock ??
    ({
      type: "signatures",
      signatures: [
        { name: "Nome do Advogado", oab: "OAB/UF XXXXX" }
      ]
    } as DocBlock);

  fechoSection.blocks = [newFechoBlock, newSignaturesBlock];
  fechoSection.titulo_literal = "Termos em que, pede deferimento.";

  if (localDataSection && localDataIndex >= 0) {
    sections.splice(localDataIndex, 1);
  }

  // Remove any lingering local_data_assinatura_oab blocks in other sections
  for (const s of sections) {
    if (!Array.isArray(s.blocks)) continue;
    s.blocks = s.blocks.filter((b) => b.block_id !== "local_data_assinatura_oab");
  }

  return result;
}

function requireApiKey(headerValue: string | undefined) {
  if (!headerValue) return false;
  return headerValue === env.APP_API_KEY;
}

function extractTextForStreaming(result: unknown): string {
  if (typeof result === "string") return result;
  if (!result || typeof result !== "object") return "";
  const doc = (result as FinalDoc).doc;
  if (!doc || !Array.isArray(doc.sections)) {
    try {
      return JSON.stringify(result);
    } catch {
      return "";
    }
  }
  const chunks: string[] = [];
  for (const section of doc.sections) {
    if (section.titulo_literal) chunks.push(section.titulo_literal);
    const blocks = Array.isArray(section.blocks) ? section.blocks : [];
    for (const block of blocks) {
      if (typeof block.text === "string" && block.text.trim()) {
        chunks.push(block.text);
      }
      if (Array.isArray(block.items) && block.items.length) {
        chunks.push(block.items.join("\n"));
      }
    }
  }
  return chunks.join("\n\n");
}

fastify.get("/health", async () => {
  return {
    ok: true,
    timestamp: new Date().toISOString()
  };
});

type JobStatus = "queued" | "running" | "done" | "error";
type JobRecord = {
  id: string;
  status: JobStatus;
  created_at: string;
  updated_at: string;
  result?: unknown;
  error?: string;
};

const jobs = new Map<string, JobRecord>();

function updateJob(id: string, patch: Partial<JobRecord>) {
  const current = jobs.get(id);
  if (!current) return;
  jobs.set(id, {
    ...current,
    ...patch,
    updated_at: new Date().toISOString()
  });
}

async function runJob(id: string, input: WorkflowRunInput) {
  updateJob(id, { status: "running" });
  try {
    const result = await runWorkflow(input);
    updateJob(id, { status: "done", result });
  } catch (err: any) {
    updateJob(id, {
      status: "error",
      error: err?.message ?? "Unexpected error"
    });
  }
}

fastify.post<{ Body: RunBody }>("/run", async (request, reply) => {
  const apiKey = request.headers["x-api-key"] as string | undefined;
  if (!requireApiKey(apiKey)) {
    reply.status(401).send({ error: "unauthorized", message: "Invalid API key" });
    return;
  }

  const parsed = runSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(400).send({
      error: "validation_error",
      message: "Invalid request body",
      issues: parsed.error.flatten()
    });
    return;
  }

  let attachments: AttachmentRecord[] = [];
  try {
    attachments = resolveAttachments(parsed.data.chat_id, parsed.data.attachment_ids);
  } catch (err: any) {
    reply.status(400).send({
      error: "validation_error",
      message: err?.message ?? "Invalid attachments"
    });
    return;
  }

  try {
    const result = await runWorkflow({
      input_as_text: parsed.data.input_as_text
    });
    reply.send(normalizeFinalJson(result));
  } catch (err: any) {
    request.log.error({ err }, "runWorkflow failed");
    reply.status(500).send({
      error: "workflow_error",
      message: err?.message ?? "Unexpected error"
    });
  }
});

fastify.post<{ Body: UploadBody }>("/upload", async (request, reply) => {
  const apiKey = request.headers["x-api-key"] as string | undefined;
  if (!requireApiKey(apiKey)) {
    reply.status(401).send({ error: "unauthorized", message: "Invalid API key" });
    return;
  }

  const parsed = uploadSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(400).send({
      error: "validation_error",
      message: "Invalid request body",
      issues: parsed.error.flatten()
    });
    return;
  }

  if (!ALLOWED_MIME_TYPES.has(parsed.data.mime_type)) {
    reply.status(400).send({
      error: "validation_error",
      message:
        "Unsupported mime_type. Supported: application/pdf, image/jpeg, image/png. Convert DOC/DOCX before upload."
    });
    return;
  }

  let fileBuffer: Buffer;
  try {
    fileBuffer = decodeBase64File(parsed.data.content_base64);
  } catch (err: any) {
    reply.status(400).send({
      error: "validation_error",
      message: err?.message ?? "Invalid content_base64"
    });
    return;
  }

  if (fileBuffer.byteLength > MAX_FILE_BYTES) {
    reply.status(400).send({
      error: "validation_error",
      message: `File too large. Maximum size is ${MAX_FILE_BYTES} bytes`
    });
    return;
  }

  try {
    const uploaded = await openai.files.create({
      file: await toFile(fileBuffer, parsed.data.filename, { type: parsed.data.mime_type }),
      purpose: "assistants"
    });
    const attachment: AttachmentRecord = {
      attachment_id: `att_${randomUUID()}`,
      chat_id: parsed.data.chat_id,
      file_id: uploaded.id,
      filename: parsed.data.filename,
      mime_type: parsed.data.mime_type,
      size_bytes: fileBuffer.byteLength,
      created_at: new Date().toISOString()
    };
    addAttachment(attachment);
    reply.send(attachment);
  } catch (err: any) {
    request.log.error({ err }, "upload failed");
    reply.status(500).send({
      error: "upload_error",
      message: err?.message ?? "Unexpected error"
    });
  }
});

fastify.post<{ Body: RunBody }>("/run-async", async (request, reply) => {
  const apiKey = request.headers["x-api-key"] as string | undefined;
  if (!requireApiKey(apiKey)) {
    reply.status(401).send({ error: "unauthorized", message: "Invalid API key" });
    return;
  }

  const parsed = runSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(400).send({
      error: "validation_error",
      message: "Invalid request body",
      issues: parsed.error.flatten()
    });
    return;
  }

  let attachments: AttachmentRecord[] = [];
  try {
    attachments = resolveAttachments(parsed.data.chat_id, parsed.data.attachment_ids);
  } catch (err: any) {
    reply.status(400).send({
      error: "validation_error",
      message: err?.message ?? "Invalid attachments"
    });
    return;
  }

  const id = randomUUID();
  const now = new Date().toISOString();
  jobs.set(id, {
    id,
    status: "queued",
    created_at: now,
    updated_at: now
  });

  setImmediate(() => {
    const workflowInput: WorkflowRunInput = {
      ...parsed.data,
      attachments
    };
    void runJob(id, workflowInput);
  });

  reply.send({ job_id: id, status: "queued" });
});

fastify.post<{ Body: RunBody }>("/run-stream", async (request, reply) => {
  const apiKey = request.headers["x-api-key"] as string | undefined;
  if (!requireApiKey(apiKey)) {
    reply.status(401).send({ error: "unauthorized", message: "Invalid API key" });
    return;
  }

  const parsed = runSchema.safeParse(request.body);
  if (!parsed.success) {
    reply.status(400).send({
      error: "validation_error",
      message: "Invalid request body",
      issues: parsed.error.flatten()
    });
    return;
  }

  let attachments: AttachmentRecord[] = [];
  try {
    attachments = resolveAttachments(parsed.data.chat_id, parsed.data.attachment_ids);
  } catch (err: any) {
    reply.status(400).send({
      error: "validation_error",
      message: err?.message ?? "Invalid attachments"
    });
    return;
  }

  reply.raw.setHeader("Content-Type", "text/event-stream; charset=utf-8");
  reply.raw.setHeader("Cache-Control", "no-cache, no-transform");
  reply.raw.setHeader("Connection", "keep-alive");
  reply.raw.setHeader("X-Accel-Buffering", "no");
  reply.hijack();

  let closed = false;
  const sendEvent = (event: string, data: Record<string, unknown>) => {
    if (closed) return;
    reply.raw.write(`event: ${event}\n`);
    reply.raw.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  request.raw.on("close", () => {
    closed = true;
  });

  const heartbeat = setInterval(() => {
    if (!closed) reply.raw.write(": keep-alive\n\n");
  }, 15000);

  try {
    sendEvent("status", { phase: "started", message: "Iniciando processamento" });

    const workflowInput: WorkflowRunInput = {
      ...parsed.data,
      attachments
    };
    const result = await runWorkflow({
      input_as_text: workflowInput.input_as_text,
      progressIntervalMs: 5000,
      onProgress: (event) => {
        if (closed) return;
        if (event.kind === "node_started") {
          sendEvent("status", {
            phase: "node_started",
            node: event.node,
            step: event.step,
            message: event.message ?? `Executando ${event.node}`
          });
          return;
        }
        if (event.kind === "node_running") {
          sendEvent("status", {
            phase: "node_running",
            node: event.node,
            step: event.step,
            elapsed_ms: event.elapsed_ms ?? 0,
            message:
              event.message ??
              `Processando ${event.node} (${Math.max(1, Math.floor((event.elapsed_ms ?? 0) / 1000))}s)`
          });
          return;
        }
        if (event.kind === "node_completed") {
          sendEvent("status", {
            phase: "node_completed",
            node: event.node,
            step: event.step,
            duration_ms: event.duration_ms ?? 0,
            message: event.message ?? `Concluido ${event.node}`
          });
          return;
        }
        if (event.kind === "node_failed") {
          sendEvent("status", {
            phase: "node_failed",
            node: event.node,
            step: event.step,
            duration_ms: event.duration_ms ?? 0,
            message: event.message ?? `Falha em ${event.node}`
          });
        }
      }
    });
    const normalized = normalizeFinalJson(result);

    sendEvent("status", { phase: "streaming_result", message: "Transmitindo resposta" });
    const text = extractTextForStreaming(normalized);
    const chunkSize = 260;
    for (let i = 0; i < text.length; i += chunkSize) {
      if (closed) break;
      sendEvent("token", { text: text.slice(i, i + chunkSize) });
      await new Promise((resolve) => setTimeout(resolve, 12));
    }

    sendEvent("done", { result: normalized });
  } catch (err: any) {
    request.log.error({ err }, "runWorkflow stream failed");
    sendEvent("error", {
      error: "workflow_error",
      message: err?.message ?? "Unexpected error"
    });
  } finally {
    clearInterval(heartbeat);
    if (!closed) reply.raw.end();
  }
});

fastify.get<{ Params: { id: string } }>("/jobs/:id", async (request, reply) => {
  const apiKey = request.headers["x-api-key"] as string | undefined;
  if (!requireApiKey(apiKey)) {
    reply.status(401).send({ error: "unauthorized", message: "Invalid API key" });
    return;
  }

  const job = jobs.get(request.params.id);
  if (!job) {
    reply.status(404).send({ error: "not_found", message: "Job not found" });
    return;
  }

  // Normalize just before returning so clients always see the standardized fecho
  if (job.status === "done") {
    reply.send({ ...job, result: normalizeFinalJson(job.result ?? "") });
    return;
  }
  reply.send(job);
});

const start = async () => {
  try {
    await fastify.listen({
      port: env.PORT,
      host: "0.0.0.0"
    });
  } catch (err) {
    fastify.log.error(err);
    process.exit(1);
  }
};

start();
