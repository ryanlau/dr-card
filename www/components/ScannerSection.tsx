"use client"

import { useEffect, useRef, useState } from "react"
import Image from "next/image"

export default function ScannerSection() {
  const [scrollPosition, setScrollPosition] = useState(0)
  const sectionRef = useRef<HTMLElement>(null)
  const [scannerComplete, setScannerComplete] = useState(false)
  const startScrollY = useRef(0)

  useEffect(() => {
    const handleScroll = () => {
      if (sectionRef.current) {
        const rect = sectionRef.current.getBoundingClientRect()
        const sectionHeight = sectionRef.current.offsetHeight
        const viewportHeight = window.innerHeight

        let progress: number

        if (!scannerComplete) {
          // Scanner phase (0-100%)
          const rawProgress = (-rect.top / (sectionHeight - viewportHeight)) * 100
          progress = Math.max(0, Math.min(100, rawProgress))

          if (progress >= 100) {
            setScannerComplete(true)
            startScrollY.current = window.scrollY
          }
        } else {
          // Text transition phase (100-200%)
          const delta = Math.max(0, window.scrollY - startScrollY.current)
          progress = 100 + delta / 10 // 1000px = 100% (1000/10=100)
        }

        setScrollPosition(progress)
      }
    }

    window.addEventListener("scroll", handleScroll, { passive: true })
    handleScroll()

    return () => {
      window.removeEventListener("scroll", handleScroll)
    }
  }, [scannerComplete])

  // First text fades out during first 10% of text transition
  const firstTextOpacity = Math.max(0, 1 - (scrollPosition - 100) / 10)

  // Second text fades in during first 10% and stays visible
  const secondTextOpacity = Math.min(1, (scrollPosition - 100) / 10)

  return (
    <section
      ref={sectionRef}
      className="relative scroll-smooth bg-theme-beige"
      style={{
        height: "calc(400vh + 1000px)", // Add 1000px for extended hold
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
                src="/aaaronnnn.jpeg"
                alt="Scan visualization"
                fill
                className="object-cover"
                priority
              />
              {/* Scanner line - only visible during scanning phase */}
              {!scannerComplete && (
                <div
                  className="absolute inset-x-0 h-8 bg-gradient-to-r from-theme-green to-theme-green/60 z-10 transition-transform duration-100 ease-linear shadow-lg"
                  style={{ top: `${scrollPosition}%` }}
                />
              )}
            </div>

            {/* Text container */}
            <div className="w-1/2 flex items-center">
              <div className="relative w-full h-48">
                {/* First text block */}
                <div
                  className="max-w-lg transition-all duration-300 ease-out absolute"
                  style={{
                    opacity: firstTextOpacity,
                    pointerEvents: firstTextOpacity > 0 ? "auto" : "none",
                  }}
                >
                  <h2 className="text-4xl font-bold mb-4 text-theme-green">Welcome to Our Innovation</h2>
                  <p className="text-lg mb-6">
                    Experience the future of grading quality sports memorabilia with our cutting-edge solutions. As you scroll, watch our
                    scanner analyze and grade the card in real time.
                  </p>
                  <button className="bg-theme-green text-theme-beige px-6 py-2 rounded-full hover:bg-theme-green/90 transition-colors">
                    Learn More
                  </button>
                </div>

                {/* Second text block */}
                <div
                  className="max-w-lg transition-all duration-300 ease-out absolute"
                  style={{
                    opacity: secondTextOpacity,
                    pointerEvents: secondTextOpacity > 0 ? "auto" : "none",
                    transform: `translateY(${Math.max(0, (scrollPosition - 150) * 0.2)}px)`,
                  }}
                >
                  <div className="bg-white/95 backdrop-blur-sm p-6 rounded-xl shadow-lg">
                    <h2 className="text-4xl font-bold mb-6 text-theme-green">Scan Complete</h2>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">PSA Grade</span>
                          <span className="font-bold text-theme-green text-xl">10</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-theme-green" style={{ width: "100%" }}></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Accuracy</span>
                          <span className="font-bold text-theme-green text-xl">95.7%</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-theme-green" style={{ width: "99.9%" }}></div>
                        </div>
                      </div>

                      <div className="space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-muted-foreground">Estimated Value</span>
                          <span className="font-bold text-theme-green text-xl">$50,000</span>
                        </div>
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <div className="h-full bg-theme-green" style={{ width: "95%" }}></div>
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

