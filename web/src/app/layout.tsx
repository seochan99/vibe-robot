import type { Metadata } from "next";
import { IBM_Plex_Mono, DM_Sans } from "next/font/google";
import FirebaseAnalytics from "@/components/FirebaseAnalytics";
import "./globals.css";

const body = DM_Sans({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

const mono = IBM_Plex_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
  weight: ["400", "500", "600"],
});

export const metadata: Metadata = {
  title: "VibeRobot! — Vibe-to-Verify for Safe Robotic Manipulation",
  description:
    "Tell a robot what to do in plain language. VibeRobot infers your intent, discovers affordances, and verifies safety before execution.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="scroll-smooth">
      <body className={`${body.variable} ${mono.variable} antialiased`}>
        <FirebaseAnalytics />
        {children}
      </body>
    </html>
  );
}
