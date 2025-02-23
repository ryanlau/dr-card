export default function TestimonialsSection() {
  const testimonials = [
    {
      name: "John Doe",
      role: "Collector & Hobbyist",
      quote: "This grading technology has completely changed the way I approach my collection—fast, fair, and incredibly accurate.",
    },
    {
      name: "Jane Smith",
      role: "Sports Memorabilia Dealer",
      quote: "With instant AI-driven grading, I can now make more informed buying and selling decisions without the long wait times.",
    },
    {
      name: "Alex Johnson",
      role: "Trading Card Investor",
      quote: "Finally, a grading system that removes the subjectivity and makes sports card investing more transparent and accessible.",
    },
  ]


  return (
    <section className="py-20 bg-theme-beige">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12 text-theme-green">What Our Clients Say</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md">
              <p className="text-muted-foreground mb-4 text-black">"{testimonial.quote}"</p>
              <div className="font-semibold text-theme-green">{testimonial.name}</div>
              <div className="text-sm text-muted-foreground">{testimonial.role}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
