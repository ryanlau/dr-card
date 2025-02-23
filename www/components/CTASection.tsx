import Link from "next/link"

export default function CTASection() {
  return (
    <section className="relative bg-theme-green text-theme-beige py-20">
      <div className="container mx-auto px-4">
        <div className="max-w-3xl">
          <h1 className="text-4xl md:text-5xl font-bold mb-4">Estimate the PSA grading.</h1>
          <p className="text-xl mb-8 text-theme-beige/90">
            Experience our revolutionary scanning technology that's changing the way grading sports cards work.
          </p>
          <Link
            href="/demo"
            className="bg-theme-beige text-theme-green px-8 py-3 rounded-full font-semibold 
                     hover:bg-white transition-colors inline-block"
          >
            Try it now
          </Link>
        </div>
      </div>
    </section>
  )
}
