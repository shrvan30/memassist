import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "MemAssist",
  description:
    "A MemGPT-style assistant that edits its own memory, on free-tier LLMs behind a failover router.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
