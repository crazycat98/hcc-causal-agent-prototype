import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "HCC Causal Agent Demo",
  description:
    "Synthetic-data HCC causal Agent prototype for research learning and portfolio demonstration.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body>{children}</body>
    </html>
  );
}
