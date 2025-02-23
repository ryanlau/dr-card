"use client"

import type React from "react"
import { useState, useCallback, useEffect } from "react"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Progress } from "@/components/ui/progress"
import DraggableQuadrilateral from "@/components/DraggableQuadrilateral"



const data = {
  "LeBronJames.jpg" : {
    "raw_price" : 3,
    "graded_price" : 85,
    "grade" : 9
  }, 
  "PatrickMahomes.jpg" : {
    "raw_price" : 45,
    "graded_price" : 90,
    "grade" : 8
  }, 
  "AaronJudge.jpg" : {
    "raw_price" : 456,
    "graded_price" : 1500,
    "grade" : 10
  }, 
}

const ImageProcessor = () => {
  const [step, setStep] = useState(0)
  const [image, setImage] = useState<string | null>(null)
  const [description, setDescription] = useState("")
  const [progress, setProgress] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  
  const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      const reader = new FileReader()
      reader.onloadend = () => {
        setImage(reader.result as string)
        setStep(1)
        setIsProcessing(true)
      }
      reader.readAsDataURL(file)
    }
    
  }, [])

const simulateProcessing = useCallback((currentStep: number) => {
  setProgress(0)
  setIsProcessing(true)
  const interval = setInterval(() => {
    setProgress((prevProgress) => {
      if (prevProgress >= 100) {
        clearInterval(interval)
        setIsProcessing(false)
        return 100
      }
      
      // Set different ranges based on step
      let min = 5
      let max = 15

      console.log(currentStep)
      if (currentStep === 1) {
        min = 10  // Larger increments for step 1
        max = 25
      } else if (currentStep === 3) {
        min = 3   // Smaller increments for step 3
        max = 8
      }

      return prevProgress + Math.floor(Math.random() * (max - min + 1) + min)
    })
  }, 200)
}, [])



  useEffect(() => {
    if (isProcessing && progress >= 100) {
      setStep((prevStep) => prevStep + 1)
      setProgress(0)
      setIsProcessing(false)
    }
  }, [progress, isProcessing])


    useEffect(() => {
    if (step === 1 || step === 3) {
        simulateProcessing(step)  // Pass current step to function
    }
    }, [step, simulateProcessing])


  const renderStep = () => {
    switch (step) {
      case 0:
        return (
          <div className="flex flex-col items-center gap-4">
            <h2 className="text-2xl font-bold">Upload an Image of a Card</h2>
            <Input type="file" accept="image/*" onChange={handleImageUpload} className="max-w-sm" />
          </div>
        )
      case 1:
        return (
          <div className="flex flex-col items-center gap-4">
            <h2 className="text-2xl font-bold">Processing Image</h2>
            <Progress value={progress} className="w-[60%]" />
          </div>
        )
      case 2:
        return (
          <div className="flex flex-col items-center gap-4">
            <h2 className="text-2xl font-bold">Card Identified</h2>
            <DraggableQuadrilateral image={image!} />
            <Input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="Card Name"
              className="w-full max-w-md"
            />
            <Button onClick={() => setStep(3)}>Confirm</Button>
          </div>
        )
      case 3:
        return (
          <div className="flex flex-col items-center gap-4">
            <h2 className="text-2xl font-bold">Final Processing</h2>
            <Progress value={progress} className="w-[60%]" />
          </div>
        )
      case 4:
        return (
          <div className="flex flex-col items-center gap-4">
            <h2 className="text-2xl font-bold">Results</h2>
            <div className="bg-green-100 p-4 rounded-lg">
              <p className="text-green-800">Processing complete!</p>
              <p className="text-green-800">Description: {description}</p>
            </div>
            <Button
              onClick={() => {
                setStep(0)
                setImage(null)
                setDescription("")
                setProgress(0)
                setIsProcessing(false)
              }}
            >
              Start Over
            </Button>
          </div>
        )
      default:
        return null
    }
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-100">
      <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">{renderStep()}</div>
    </div>
  )
}

export default function Page() {
  return (
    <div className="font-sans">
      <ImageProcessor />
    </div>
  )
}

