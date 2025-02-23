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
  data: any
  showPoints: boolean
}

const DraggableQuadrilateral: React.FC<DraggableQuadrilateralProps> = ({ image, data, showPoints }) => {
  const startingPoints = data.points
  const containerRef = useRef<HTMLDivElement>(null)
  const [imageSize, setImageSize] = useState({ width: 0, height: 0 })
  const [isDragging, setIsDragging] = useState<string | null>(null)
  const [points, setPoints] = useState<Point[]>([
    { id: "tl", x: 100, y: 100 },
    { id: "tr", x: 300, y: 100 },
    { id: "br", x: 300, y: 300 },
    { id: "bl", x: 100, y: 300 },
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
      const defaultWidth = Math.min(300, imgElement.naturalWidth * 0.5)
      const defaultHeight = Math.min(300, imgElement.naturalHeight * 0.5)
      const startX = (imgElement.naturalWidth - defaultWidth) / 2
      const startY = (imgElement.naturalHeight - defaultHeight) / 2

      if (startingPoints) {
        setPoints(startingPoints)
      } else {
        setPoints([
          { id: "tl", x: startX, y: startY },
          { id: "tr", x: startX + defaultWidth, y: startY },
          { id: "br", x: startX + defaultWidth, y: startY + defaultHeight },
          { id: "bl", x: startX, y: startY + defaultHeight },
        ])
      }
    }
  }, [image, startingPoints])

  const handleMouseDown = (pointId: string) => (e: React.MouseEvent) => {
    e.preventDefault()
    const point = points.find((point) => point.id === pointId)
    console.log(point!.id)
    console.log(point!.x)
    console.log(point!.y)
    
    setIsDragging(pointId)
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging || !containerRef.current) return

    const container = containerRef.current.getBoundingClientRect()
    const scale = container.width / imageSize.width

    const x = Math.min(Math.max(0, (e.clientX - container.left) / scale), imageSize.width)
    const y = Math.min(Math.max(0, (e.clientY - container.top) / scale), imageSize.height)

    setPoints((currentPoints) => currentPoints.map((point) => (point.id === isDragging ? { ...point, x, y } : point)))
  }

  const handleMouseUp = () => {
    setIsDragging(null)
  }

  const createPath = () => {
    return `M ${points[0].x},${points[0].y} 
            L ${points[1].x},${points[1].y} 
            L ${points[2].x},${points[2].y} 
            L ${points[3].x},${points[3].y} Z`
  }

  if (!imageSize.width || !imageSize.height) {
    return <div>Loading...</div>
  }

  // Calculate a fixed radius that scales with the viewport
  const pointRadius = Math.min(imageSize.width, imageSize.height) * 0.02 // 2% of the smaller dimension

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

        <svg
          className="absolute top-0 left-0 w-full h-full pointer-events-none"
          viewBox={`0 0 ${imageSize.width} ${imageSize.height}`}
          preserveAspectRatio="none"
        >
          {/* Base quadrilateral */}
          <path
            d={createPath()}
            fill="rgba(59, 130, 246, 0.2)"
            stroke="rgb(59, 130, 246)"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />

          {/* Ripple effects - multiple paths with different animations */}
          <path
            d={createPath()}
            fill="none"
            stroke="rgb(59, 130, 246)"
            strokeWidth="3"
            className="animate-ripple-1"
            strokeDasharray="10"
            opacity="0"
            vectorEffect="non-scaling-stroke"
          />
          <path
            d={createPath()}
            fill="none"
            stroke="rgb(59, 130, 246)"
            strokeWidth="3"
            className="animate-ripple-2"
            strokeDasharray="10"
            opacity="0"
            vectorEffect="non-scaling-stroke"
          />
          <path
            d={createPath()}
            fill="none"
            stroke="rgb(59, 130, 246)"
            strokeWidth="3"
            className="animate-ripple-3"
            strokeDasharray="10"
            opacity="0"
            vectorEffect="non-scaling-stroke"
          />
        </svg>

        {/* Draggable Points */}
        {showPoints && points.map((point) => (
          <div
            key={point.id}
            className="absolute cursor-move bg-white border-2 border-blue-500 rounded-full hover:bg-blue-100 transition-colors"
            style={{
              left: `${(point.x / imageSize.width) * 100}%`,
              top: `${(point.y / imageSize.height) * 100}%`,
              transform: "translate(-50%, -50%)",
              width: "32px",
              height: "32px",
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

