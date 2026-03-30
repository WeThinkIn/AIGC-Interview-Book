const menuToggle = document.getElementById("menu-toggle");
const siteNav = document.getElementById("site-nav");
const footerYear = document.getElementById("footer-year");
const header = document.querySelector(".site-header");
const particleField = document.getElementById("particle-field");

if (menuToggle && siteNav) {
  menuToggle.addEventListener("click", () => {
    const expanded = menuToggle.getAttribute("aria-expanded") === "true";
    menuToggle.setAttribute("aria-expanded", String(!expanded));
    siteNav.classList.toggle("is-open", !expanded);
  });

  siteNav.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", () => {
      menuToggle.setAttribute("aria-expanded", "false");
      siteNav.classList.remove("is-open");
    });
  });
}

if (footerYear) {
  footerYear.textContent = `© ${new Date().getFullYear()} 三年面试五年模拟`;
}

window.addEventListener(
  "scroll",
  () => {
    header?.classList.toggle("is-scrolled", window.scrollY > 10);
  },
  { passive: true },
);

const revealObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add("is-visible");
        revealObserver.unobserve(entry.target);
      }
    });
  },
  {
    threshold: 0.18,
    rootMargin: "0px 0px -40px 0px",
  },
);

document.querySelectorAll(".reveal").forEach((element) => revealObserver.observe(element));

if (particleField) {
  const particleCount = window.innerWidth < 780 ? 18 : 28;

  for (let index = 0; index < particleCount; index += 1) {
    const particle = document.createElement("span");
    particle.style.setProperty("--size", `${Math.random() * 3 + 1}px`);
    particle.style.setProperty("--top", `${Math.random() * 100}%`);
    particle.style.setProperty("--left", `${Math.random() * 100}%`);
    particle.style.setProperty("--duration", `${Math.random() * 8 + 10}s`);
    particle.style.setProperty("--delay", `${Math.random() * -12}s`);
    particle.style.setProperty("--drift-x", `${Math.random() * 40 - 20}px`);
    particleField.appendChild(particle);
  }
}
