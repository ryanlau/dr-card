"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import Image from "next/image"

type Point = {
  id: string
  x: number
  y: number
}

interface DraggableQuadrilateralProps {
  image: string
}

const DraggableQuadrilateral: React.FC<DraggableQuadrilateralProps> = ({ image }) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [isDragging, setIsDragging] = useState<string | null>(null)
  const [points, setPoints] = useState<Point[]>([
    { id: "tl", x: 100, y: 100 }, // Top-left
    { id: "tr", x: 300, y: 100 }, // Top-right
    { id: "br", x: 300, y: 300 }, // Bottom-right
    { id: "bl", x: 100, y: 300 }, // Bottom-left
  ])

  // Get image dimensions on load
  useEffect(() => {
    const imgElement = document.createElement("img")
    imgElement.src = image
    imgElement.onload = () => {
      setImageSize({
        width: imgElement.naturalWidth,
        height: imgElement.naturalHeight,
      })
      // Initialize points at reasonable default positions
      const defaultWidth = Math.min(300, imgElement.naturalWidth * 0.5)
      const defaultHeight = Math.min(300, imgElement.naturalHeight * 0.5)
      const startX = (imgElement.naturalWidth - defaultWidth) / 2
      const startY = (imgElement.naturalHeight - defaultHeight) / 2

      setPoints([
        { id: "tl", x: startX, y: startY },
        { id: "tr", x: startX + defaultWidth, y: startY },
        { id: "br", x: startX + defaultWidth, y: startY + defaultHeight },
        { id: "bl", x: startX, y: startY + defaultHeight },
      ])
    }
  }, [image])

  const handleMouseDown = (pointId: string) => (e: React.MouseEvent) => {
    e.preventDefault()
    setIsDragging(pointId)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !containerRef.current) return

    const container = containerRef.current.getBoundingClientRect()
    const scale = container.width / imageSize.width // Calculate scale factor

    // Adjust coordinates based on scale and container position
    const x = Math.min(Math.max(0, (e.clientX - container.left) / scale), imageSize.width)
    const y = Math.min(Math.max(0, (e.clientY - container.top) / scale), imageSize.height)

    setPoints((currentPoints) => currentPoints.map((point) => (point.id === isDragging ? { ...point, x, y } : point)))
  }

  const handleMouseUp = () => {
    setIsDragging(null)
  }

  // Create the SVG path for the quadrilateral
  const createPath = () => {
    return `M ${points[0].x},${points[0].y} 
            L ${points[1].x},${points[1].y} 
            L ${points[2].x},${points[2].y} 
            L ${points[3].x},${points[3].y} Z`
  }

  if (!imageSize.width || !imageSize.height) {
    return <div>Loading...</div>
  }

  return (
    <div className="flex flex-col items-center gap-4">
      <div
        ref={containerRef}
        className="relative select-none"
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
      >
        <Image
          src={image || "/placeholder.svg"}
          alt="Uploaded image"
          width={imageSize.width}
          height={imageSize.height}
          className="rounded-lg"
          unoptimized
        />

        {/* SVG Overlay for the quadrilateral */}
        <svg
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          preserveAspectRatio="none"
        >
          <path d={createPath()} fill="rgba(59, 130, 246, 0.2)" stroke="rgb(59, 130, 246)" strokeWidth="2" />

          {/* SVG Points */}
          {points.map((point) => (
            <circle
              key={point.id}
              cx={point.x}
              cy={point.y}
              r="24"
              fill="white"
              stroke="rgb(59, 130, 246)"
              strokeWidth="2"
              className="pointer-events-none"
            />
          ))}
        </svg>

        {/* Invisible Draggable Areas */}
        {points.map((point) => (
          <div
            key={point.id}
            className="absolute w-12 h-12 cursor-move"
            style={{
              left: `${(point.x / imageSize.width) * 100}%`,
              top: `${(point.y / imageSize.height) * 100}%`,
              transform: "translate(-50%, -50%)",
              touchAction: "none",
            }}
            onMouseDown={handleMouseDown(point.id)}
          />
        ))}
      </div>
    </div>
  )
}

export default DraggableQuadrilateral

