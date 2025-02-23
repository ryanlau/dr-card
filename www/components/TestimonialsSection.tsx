export default function TestimonialsSection() {
  const testimonials = [
    {
      name: "John Doe",
      role: "CEO, TechCorp",
      quote: "This scanner technology has transformed our data processing capabilities.",
    },
    {
      name: "Jane Smith",
      role: "CTO, InnovateCo",
      quote: "We've seen a 50% increase in efficiency since implementing this solution.",
    },
    {
      name: "Alex Johnson",
      role: "Data Scientist, AnalyticsPro",
      quote: "The accuracy and speed of this scanner are unparalleled in the industry.",
    },
  ]

  return (
    <section className="py-20 bg-theme-beige">
      <div className="container mx-auto px-4">
        <h2 className="text-3xl font-bold text-center mb-12 text-theme-green">What Our Clients Say</h2>
        <div className="grid md:grid-cols-3 gap-8">
          {testimonials.map((testimonial, index) => (
            <div key={index} className="bg-white p-6 rounded-lg shadow-md">
              <p className="text-muted-foreground mb-4">"{testimonial.quote}"</p>
              <div className="font-semibold text-theme-green">{testimonial.name}</div>
              <div className="text-sm text-muted-foreground">{testimonial.role}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  )
}
