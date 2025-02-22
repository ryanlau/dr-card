import type { Metadata } from 'next'
import { Public_Sans } from 'next/font/google'

import './globals.css'

export const metadata: Metadata = {
  title: 'v0 App',
  description: 'Created with v0',
  generator: 'v0.dev',
}

const public_sans = Public_Sans({
  display: 'swap',
  subsets: ['latin'],
  variable: '--font-public-sans',
})

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className={`${public_sans.variable}`}>
      <body>{children}</body>
    </html>
  )
}
