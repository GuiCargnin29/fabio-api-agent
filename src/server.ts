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
    reply.send(result);
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
