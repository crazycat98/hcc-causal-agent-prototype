import { z } from "zod";

export const causalFeatureNames = [
  "tumor_size_cm",
  "afp_ng_ml",
  "alt_u_l",
  "ast_u_l",
  "bilirubin_umol_l",
  "albumin_g_l",
  "platelet_10e9_l",
  "portal_vein_invasion",
  "radiomics_entropy",
  "radiomics_glcm_contrast",
] as const;

export type CausalFeatureName = (typeof causalFeatureNames)[number];

export const hccFeatureSchema = z
  .object({
    tumor_size_cm: z
      .number()
      .min(0.5)
      .max(20)
      .describe("Synthetic tumor diameter in centimeters."),
    afp_ng_ml: z
      .number()
      .min(0.5)
      .max(50000)
      .describe("Synthetic alpha-fetoprotein value."),
    alt_u_l: z.number().min(5).max(500).describe("Synthetic ALT value."),
    ast_u_l: z.number().min(5).max(500).describe("Synthetic AST value."),
    bilirubin_umol_l: z
      .number()
      .min(2)
      .max(150)
      .describe("Synthetic total bilirubin value."),
    albumin_g_l: z.number().min(15).max(55).describe("Synthetic albumin value."),
    platelet_10e9_l: z
      .number()
      .min(20)
      .max(500)
      .describe("Synthetic platelet count."),
    portal_vein_invasion: z
      .number()
      .int()
      .min(0)
      .max(1)
      .describe("Synthetic binary portal vein invasion flag, 0 or 1."),
    radiomics_entropy: z
      .number()
      .min(2)
      .max(8)
      .describe("Synthetic radiomics entropy feature."),
    radiomics_glcm_contrast: z
      .number()
      .min(10)
      .max(250)
      .describe("Synthetic GLCM contrast feature."),
  })
  .strict();

export const partialHccFeatureSchema = hccFeatureSchema.partial().strict();

export type HccFeatures = z.infer<typeof hccFeatureSchema>;
export type PartialHccFeatures = z.infer<typeof partialHccFeatureSchema>;

export function listMissingFeatures(features: PartialHccFeatures): CausalFeatureName[] {
  return causalFeatureNames.filter((name) => features[name] === undefined);
}

export function assertCompleteFeatures(features: PartialHccFeatures): HccFeatures {
  return hccFeatureSchema.parse(features);
}

