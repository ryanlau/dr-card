import CTASection from "@/components/CTASection"
import ScannerSection from "@/components/ScannerSection"
import TestimonialsSection from "@/components/TestimonialsSection"
import Footer from "@/components/Footer"

export default function Home() {
  return (
    <div className="relative font-sans">
      <CTASection />
      <ScannerSection />
      <TestimonialsSection />
      <Footer />
    </div>
  )
}

