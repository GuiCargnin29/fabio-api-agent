import "dotenv/config";
import Fastify from "fastify";
import cors from "@fastify/cors";
import helmet from "@fastify/helmet";
import rateLimit from "@fastify/rate-limit";
import { z } from "zod";
import { runWorkflow } from "./agent.js";
import { randomUUID } from "node:crypto";

const envSchema = z.object({
  OPENAI_API_KEY: z.string().min(1, "OPENAI_API_KEY is required"),
  APP_API_KEY: z.string().min(16, "APP_API_KEY must be at least 16 chars"),
  PORT: z.coerce.number().int().positive().default(3000),
  LOG_LEVEL: z.string().optional(),
  NODE_ENV: z.string().optional()
});

const env = envSchema.parse(process.env);

const fastify = Fastify({
  logger: {
    level: env.LOG_LEVEL ?? "info"
  },
  bodyLimit: 1_000_000
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

const runSchema = z.object({
  input_as_text: z.string().min(1, "input_as_text is required")
});

type RunBody = z.infer<typeof runSchema>;

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

async function runJob(id: string, input: RunBody) {
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

  try {
    const result = await runWorkflow(parsed.data);
    reply.send(normalizeFinalJson(result));
  } catch (err: any) {
    request.log.error({ err }, "runWorkflow failed");
    reply.status(500).send({
      error: "workflow_error",
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

  const id = randomUUID();
  const now = new Date().toISOString();
  jobs.set(id, {
    id,
    status: "queued",
    created_at: now,
    updated_at: now
  });

  setImmediate(() => {
    void runJob(id, parsed.data);
  });

  reply.send({ job_id: id, status: "queued" });
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
  if (job.status === "done" && job.result) {
    reply.send({ ...job, result: normalizeFinalJson(job.result) });
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
