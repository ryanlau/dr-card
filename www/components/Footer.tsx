export default function Footer() {
  return (
    <footer className="bg-theme-green text-theme-beige py-8">
      <div className="container mx-auto px-4">
        <div className="grid md:grid-cols-3 gap-8">
          <div>
            <h3 className="text-lg font-semibold mb-4">About Us</h3>
            <p className="text-theme-beige/80">
            We are innovators in AI-driven sports card grading, making the hobby more accessible, transparent, and fair for collectors everywhere.            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <a href="#" className="text-theme-beige/80 hover:text-theme-beige transition-colors">
                  Home
                </a>
              </li>
              <li>
                <a href="#" className="text-theme-beige/80 hover:text-theme-beige transition-colors">
                  Products
                </a>
              </li>
              <li>
                <a href="#" className="text-theme-beige/80 hover:text-theme-beige transition-colors">
                  About
                </a>
              </li>
              <li>
                <a href="#" className="text-theme-beige/80 hover:text-theme-beige transition-colors">
                  Contact
                </a>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact Us</h3>
            <p className="text-theme-beige/80">123 Tech Street, Atlanta, GA 30332</p>
            <p className="text-theme-beige/80">Phone: (123) 456-7890</p>
            <p className="text-theme-beige/80">Email: info@graderef.com</p>
          </div>
        </div>
        <div className="mt-8 pt-8 border-t border-theme-beige/20 text-center text-theme-beige/80">
          © 2025 GradeRef Inc. All rights reserved.
        </div>
      </div>
    </footer>
  )
}

