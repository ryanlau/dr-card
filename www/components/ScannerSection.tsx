"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"

export default function ScannerSection() {
  const [scrollPosition, setScrollPosition] = useState(0)
  const sectionRef = useRef<HTMLElement>(null)
  const [scannerComplete, setScannerComplete] = useState(false)

  useEffect(() => {
    const handleScroll = () => {
      if (sectionRef.current) {
        const rect = sectionRef.current.getBoundingClientRect()
        const sectionHeight = sectionRef.current.offsetHeight
        const viewportHeight = window.innerHeight

        // Calculate raw scroll progress
        const rawProgress = (-rect.top / (sectionHeight - viewportHeight)) * 100

        // Control the scroll progression
        let progress
        if (rawProgress <= 100) {
          // Normal scroll until scanner reaches bottom
          progress = rawProgress
        } else if (rawProgress <= 150) {
          // Create a "sticky" section for text transition
          // Keep progress at 100-115 range until raw progress reaches 150
          progress = 100 + (rawProgress - 100) / 5
        } else {
          // Resume normal scrolling after transition
          progress = rawProgress
        }

        progress = Math.max(0, Math.min(300, progress))
        setScrollPosition(progress)

        if (progress >= 100 && !scannerComplete) {
          setScannerComplete(true)
        }
      }
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener("scroll", handleScroll)
    }
  }, [scannerComplete])

  // First text block animations
  const firstTextOpacity = scrollPosition <= 100 ? 1 : Math.max(0, 1 - (scrollPosition - 100) / 7)

  // Second text block animations
  const secondTextOpacity = scrollPosition < 107 ? 0 : Math.min(1, (scrollPosition - 107) / 7)

  // Keep second text centered until much later
  const secondTextTransform = scrollPosition > 200 ? `translateY(${-(scrollPosition - 200) * 0.5}px)` : "translateY(0)"

  return (
    <section
      ref={sectionRef}
      className="relative scroll-smooth"
      style={{
        height: "800vh",
        // Force a new stacking context
        isolation: "isolate",
      }}
      id="scanner"
    >
      <div className="sticky top-0 h-screen overflow-hidden">
        <div className="absolute inset-0 flex items-center justify-center max-w-7xl mx-auto px-4 md:px-8">
          <div className="flex w-full gap-8 md:gap-12">
            {/* Image container */}
            <div className="w-1/2 relative aspect-[3/4] rounded-xl overflow-hidden">
              <Image
                src="/placeholder.svg?height=800&width=600"
                alt="Descriptive alt text"
                fill
                className="object-cover"
                priority
              />
              {/* Scanner line */}
              <div
                className="absolute inset-x-0 h-1 bg-gradient-to-r from-primary to-primary-foreground z-10 transition-transform duration-100 ease-linear shadow-lg"
                style={{ top: `${Math.min(100, scrollPosition)}%` }}
              />
            </div>

            {/* Text container */}
            <div className="w-1/2 flex items-center">
              <div className="relative w-full">
                {/* First text block */}
                <div
                  className="max-w-lg transition-all duration-300 ease-out absolute"
                  style={{
                    opacity: firstTextOpacity,
                    transform: "translateY(0)",
                  }}
                >
                  <h2 className="text-4xl font-bold mb-4">Welcome to Our Innovation</h2>
                  <p className="text-lg mb-6">
                    Experience the future of technology with our cutting-edge solutions. As you scroll, watch our
                    scanner analyze and process information in real-time.
                  </p>
                  <button className="bg-primary text-primary-foreground px-6 py-2 rounded-full hover:bg-primary/90 transition-colors">
                    Learn More
                  </button>
                </div>

                {/* Second text block */}
                <div
                  className="max-w-lg transition-all duration-300 ease-out absolute"
                  style={{
                    transform: secondTextTransform,
                    opacity: secondTextOpacity,
                  }}
                >
                  <div className="bg-background/95 backdrop-blur-sm p-6 rounded-xl shadow-lg">
                    <h2 className="text-4xl font-bold mb-6">Scan Complete</h2>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Processing Speed</span>
                          <span className="font-bold text-primary text-xl">10x Faster</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-primary" style={{ width: "100%" }}></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Accuracy Rate</span>
                          <span className="font-bold text-primary text-xl">99.9%</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-primary" style={{ width: "99.9%" }}></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Data Points</span>
                          <span className="font-bold text-primary text-xl">1M+</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-primary" style={{ width: "95%" }}></div>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}

