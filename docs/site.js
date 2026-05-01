const menuToggle = document.getElementById("menu-toggle");
const siteNav = document.getElementById("site-nav");
const footerYear = document.getElementById("footer-year");
const header = document.querySelector(".site-header");
const particleField = document.getElementById("particle-field");
const highlightsRoot = document.getElementById("highlights-root");
const githubStars = document.getElementById("github-stars");

const formatStarCount = (count) => {
  if (!Number.isFinite(count)) {
    return "Star";
  }

  if (count >= 1000) {
    return `${(Math.round(count / 100) / 10).toFixed(1)}+k`;
  }

  return String(count);
};

const renderGithubStars = async () => {
  if (!githubStars || !window.fetch) {
    return;
  }

  try {
    const response = await fetch("https://api.github.com/repos/WeThinkIn/AIGC-Interview-Book", {
      headers: {
        Accept: "application/vnd.github+json",
      },
    });

    if (!response.ok) {
      throw new Error("GitHub API unavailable");
    }

    const repo = await response.json();
    githubStars.textContent = formatStarCount(repo.stargazers_count);
  } catch {
    githubStars.textContent = "Star";
  }
};

const renderTagList = (items = [], className = "frontier-pill") =>
  items
    .map((item) => {
      const label = typeof item === "string" ? item : item.label;
      const tone = typeof item === "string" ? "neutral" : item.tone ?? "neutral";

      return `<span class="${className} is-${tone}">${label}</span>`;
    })
    .join("");

const renderFrontierHighlights = () => {
  const data = window.frontierHighlightsData;

  if (!highlightsRoot || !data) {
    return;
  }

  const themeCards = data.themes
    .map(
      (theme) => `
        <article class="frontier-card frontier-card-${theme.key} reveal">
          <div class="frontier-card-head">
            <div class="frontier-card-meta">
              <span class="frontier-card-tag">${theme.badge}</span>
              <span class="frontier-card-slash">${theme.shortLabel}</span>
            </div>
            <span class="frontier-card-count">${theme.count}</span>
          </div>
          <h3>${theme.title}</h3>
          <p class="frontier-card-copy">${theme.descriptionHtml}</p>
          <div class="frontier-keywords">
            ${renderTagList(theme.spotlight, "frontier-keyword")}
          </div>
          <div class="frontier-list">
            ${theme.items
              .map(
                (item) => `
                  <article class="frontier-item">
                    <span class="frontier-item-index">${item.index}</span>
                    <div class="frontier-item-body">
                      <span class="frontier-item-overline">${item.overline}</span>
                      <h4>${item.title}</h4>
                      <p>${item.summaryHtml}</p>
                      ${
                        item.resources?.length
                          ? `<div class="frontier-sub-links">
                              ${item.resources
                                .map(
                                  (resource) => `
                                    <a href="${resource.url}" target="_blank" rel="noreferrer">${resource.label}</a>
                                  `,
                                )
                                .join("")}
                            </div>`
                          : ""
                      }
                    </div>
                    <a class="frontier-link" href="${item.url}" target="_blank" rel="noreferrer">原文</a>
                  </article>
                `,
              )
              .join("")}
          </div>
        </article>
      `,
    )
    .join("");

  const benefitCards = data.benefits.items
    .map(
      (item) => `
        <article class="frontier-benefit-item">
          <strong>${item.title}</strong>
          <p>${item.text}</p>
        </article>
      `,
    )
    .join("");

  highlightsRoot.innerHTML = `
    <div class="section-heading reveal">
      <p class="eyebrow">${data.header.eyebrow}</p>
      <h2>${data.header.title}</h2>
      <p>${data.header.description}</p>
    </div>

    <div class="frontier-hero reveal">
      <div class="frontier-hero-copy">
        <span class="band-label">${data.hero.label}</span>
        <h3>${data.hero.titleHtml}</h3>
        <p>${data.hero.description}</p>
        <div class="frontier-pill-row">
          ${renderTagList(data.hero.pills)}
        </div>
      </div>
      <div class="frontier-summary">
        ${data.hero.metrics
          .map(
            (metric) => `
              <div class="frontier-summary-card">
                <span>${metric.label}</span>
                <strong>${metric.valueHtml}</strong>
              </div>
            `,
          )
          .join("")}
      </div>
    </div>

    <div class="frontier-grid">
      ${themeCards}
    </div>

    <div class="frontier-bottom">
      <div class="frontier-benefits reveal">
        <span class="band-label">${data.benefits.label}</span>
        <h3>${data.benefits.titleHtml}</h3>
        <div class="frontier-benefit-grid">
          ${benefitCards}
        </div>
      </div>

      <aside class="frontier-cta reveal">
        <span class="image-tag">${data.cta.label}</span>
        <h3>${data.cta.titleHtml}</h3>
        <p>${data.cta.description}</p>
        <div class="frontier-cta-pills">
          ${renderTagList(data.cta.bullets)}
        </div>
        <a class="button primary" href="${data.cta.buttonUrl}" target="_blank" rel="noreferrer">${data.cta.buttonText}</a>
        <p class="frontier-footnote">${data.cta.footnote}</p>
      </aside>
    </div>
  `;
};

renderFrontierHighlights();
renderGithubStars();

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
