import Link from "next/link"

export default function CTASection() {
  return (
    <section className="relative bg-primary text-primary-foreground py-20">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Transform Your Data Processing</h1>
          <p className="text-xl mb-8 text-primary-foreground/90">
            Experience our revolutionary scanning technology that's changing how businesses handle information.
          </p>
          <Link
            href="/demo"
            className="bg-background text-foreground px-8 py-3 rounded-full font-semibold 
                     hover:bg-background/90 transition-colors inline-block"
          >
            Try it now
          </Link>
        </div>
      </div>
    </section>
  )
}

