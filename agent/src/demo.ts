import type { HccFeatures } from "./features.js";
import { runHccAgent } from "./runner.js";

const demoFeatures: HccFeatures = {
  tumor_size_cm: 6.2,
  afp_ng_ml: 420,
  alt_u_l: 61,
  ast_u_l: 72,
  bilirubin_umol_l: 24,
  albumin_g_l: 36,
  platelet_10e9_l: 128,
  portal_vein_invasion: 1,
  radiomics_entropy: 5.4,
  radiomics_glcm_contrast: 112,
};

async function waitForMlService() {
  const healthUrl = process.env.ML_HEALTH_URL ?? "http://127.0.0.1:8001/health";
  const attempts = 40;

  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    try {
      const response = await fetch(healthUrl, {
        signal: AbortSignal.timeout(1000),
      });
      if (response.ok) {
        return;
      }
    } catch {
      // Keep waiting; shap/numba imports can make the Python service cold start slowly.
    }

    await new Promise((resolve) => setTimeout(resolve, 500));
  }

  throw new Error(
    `ML service is not ready at ${healthUrl}. Start it with: npm.cmd run m1:serve`,
  );
}

await waitForMlService();

const result = await runHccAgent({
  patientId: "demo-001",
  features: demoFeatures,
});

console.log("M5 Agent trace");
console.log(
  JSON.stringify(
    {
      finishReason: result.finishReason,
      steps: result.steps,
      toolCalls: result.toolCalls.map((call) => call.toolName),
    },
    null,
    2,
  ),
);
console.log("");
console.log(result.text);
