"use client"

import type React from "react"
import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import DraggableQuadrilateral from "@/components/DraggableQuadrilateral"



const data = {
	"LeBronJames.jpg": {
		"raw_price": 1000,
		"graded_price": 1500,
		"grade": 9,
		"points": [
			{ id: "tl", x: 251.3, y: 18.42 },
			{ id: "tr", x: 1443.42, y: 28.94 },
			{ id: "br", x: 1364.47, y: 1563.15 },
			{ id: "bl", x: 288.15, y: 1565.78 }
		],
		"cardName": "Topps - 2005-06 LeBron James - #200",
		"summary": "This card is in great condition, with no visible scratches or marks. The edges are sharp and most corners are crisp. The card is centered well and the surface is clean.",
		"value_increase": "By grading this card, you could increase its value by $500."
	},
	"PatrickMahomes.jpg": {
		"raw_price": 45,
		"graded_price": 90,
		"grade": 8,
		"points": [
			{ id: "tl", x: 61.5, y: 202 },
			{ id: "tr", x: 408, y: 197 },
			{ id: "br", x: 414, y: 696 },
			{ id: "bl", x: 63, y: 694 }
		],
		"cardName": "National Treasures 2023 - Patrick Mahomes Holo Gold",
  // this card is in good condition
		"summary": "This card is in good condition, with a few visible scratches and marks. The edges are slightly worn and the corners are slightly rounded. The card is slightly off-center and the surface has a few scratches.",
		"value_increase": "By grading this card, you could increase its value by $45."
	},
	"AaronJudge.png": {
		"raw_price": 100,
		"graded_price": 900,
		"grade": 10,
    "points": [
			{ id: "tl", x: 21.89, y: 49.04 },
			{ id: "tr", x: 1037.85, y: 59.55 },
			{ id: "br", x: 1020, y: 1465 },
			{ id: "bl", x: 37, y: 1462.62 }
		],
		"cardName": "2022 Panini Donruss Optic Neon - Aaron Judge",
		"summary": "This card is in pristine condition, with no visible scratches or marks. The edges are sharp and the corners are crisp. The card is centered well and the surface is clean.",
		"value_increase": "By grading this card, you could increase its value by $800."
	},

}

const ImageProcessor = () => {
	const [step, setStep] = useState(0)
	const [image, setImage] = useState<string | null>(null)
	const [fileName, setFileName] = useState<string>("")
	const [description, setDescription] = useState("")
	const [progress, setProgress] = useState(0)
	const [isProcessing, setIsProcessing] = useState(false)

	const handleImageUpload = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
		const file = event.target.files?.[0]
		setFileName(file!.name)
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
						<h2 className="text-2xl font-bold">Upload an Image of a card</h2>
						<Input type="file" accept="image/*" onChange={handleImageUpload} className="max-w-sm cursor-pointer" />
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
						{/* 
            // @ts-ignore */}
						<DraggableQuadrilateral showPoints={true} image={image!} data={data[fileName]} />
						<Input
							// @ts-ignore
							value={data[fileName]["cardName"]}
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
						<h2 className="text-2xl font-bold">Assessing your card</h2>
						<Progress value={progress} className="w-[60%]" />
					</div>
				)
			case 4:
				return (
					<div className="flex flex-col gap-4">
						<h2 className="text-2xl font-bold">Results</h2>
						<div className="rounded-lg">
              <div className="flex grid-cols-5 gap-8">
                <div className="col-span-2">
                  {/* 
                  // @ts-ignore */}
                  <DraggableQuadrilateral showPoints={false} image={image!} data={data[fileName]} />
                </div>

                <div className="col-span-3">
                  {/* 
                  // @ts-ignore */}
                  <p className="text-black">{data[fileName]["summary"]}</p>
                  <br></br>

                  {/* 
                  // @ts-ignore */}
                  <p className="text-black">Expected Grade: PSA {data[fileName]["grade"]}</p>

                  <br></br>

                  {/* 
                  // @ts-ignore */}
                  <p className="text-black">Raw Price: ${data[fileName]["raw_price"]}</p>
                  {/* 
                  // @ts-ignore */}
                  <p className="text-black">Graded Price: ${data[fileName]["graded_price"]}</p>

                  {/* 
                  // @ts-ignore */}
                  <p className="text-black">{data[fileName]["value_increase"]}</p>
                  </div>
                </div>
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
							Scan next card
						</Button>
					</div>
				)
			default:
				return null
		}
	}


	return (
		<div className="min-h-screen flex items-center justify-center bg-gray-100">
      {step == 4 && (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-4xl">{renderStep()}</div>
      )}

      {step != 4 && (
        <div className="bg-white p-8 rounded-xl shadow-lg w-full max-w-2xl">{renderStep()}</div>
        )}
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

