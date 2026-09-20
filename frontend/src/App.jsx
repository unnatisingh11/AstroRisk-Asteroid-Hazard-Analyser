import { useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL;

function App() {
  const [search, setSearch] = useState("");
  const [results, setResults] = useState([]);
  const [selectedAsteroid, setSelectedAsteroid] = useState(null);

  const [page, setPage] = useState("search");

  const [loading, setLoading] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState("");

  // SEARCH ASTEROIDS
  const handleSearch = async () => {
    const query = search.trim();

    if (!query) {
      setError("Enter an asteroid name or designation.");
      return;
    }

    setSearching(true);
    setError("");
    setResults([]);
    setSelectedAsteroid(null);

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/asteroids/search?q=${encodeURIComponent(query)}`
      );

      if (!response.ok) {
        throw new Error("Search failed");
      }

      const data = await response.json();

      if (!data.results || data.results.length === 0) {
        setError("No matching asteroid found.");
      } else {
        setResults(data.results);
      }
    } catch (error) {
      setError(
        "Could not connect to AstroRisk API. Make sure FastAPI is running."
      );
    } finally {
      setSearching(false);
    }
  };

  // ANALYZE ASTEROID
  const handleAnalyze = async (spkid) => {
    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/asteroids/${spkid}`
      );

      if (!response.ok) {
        throw new Error("Analysis failed");
      }

      const data = await response.json();

      setSelectedAsteroid(data);

      // Move to dashboard
      setPage("dashboard");

      // Scroll to top
      window.scrollTo({
        top: 0,
        behavior: "smooth",
      });
    } catch (error) {
      setError("Unable to analyze this asteroid.");
    } finally {
      setLoading(false);
    }
  };

  // ENTER KEY SEARCH
  const handleKeyDown = (event) => {
    if (event.key === "Enter") {
      handleSearch();
    }
  };

  // BACK TO SEARCH
  const handleBack = () => {
    setPage("search");
    setSelectedAsteroid(null);
    setError("");

    window.scrollTo({
      top: 0,
      behavior: "smooth",
    });
  };

  return (
    <div className="app">

      {/* HEADER */}

      <header className="header">
        <div
          className="logo"
          onClick={handleBack}
        >
          <span className="logo-mark">
            A
          </span>

          <div>
            <h1>AstroRisk</h1>
            <p>Asteroid Hazard Analyzer</p>
          </div>
        </div>
      </header>


      {/* ================= SEARCH PAGE ================= */}

      {page === "search" && (
        <main className="main">

          <section className="hero">

            <p className="eyebrow">
              NASA/JPL DATA • MACHINE LEARNING
            </p>

            <h2>
              Explore asteroid
              <br />
              hazard indicators.
            </h2>

            <p className="description">
              AstroRisk analyzes known near-Earth asteroid
              data using machine learning to classify
              Potentially Hazardous Asteroid status.
            </p>


            {/* SEARCH BAR */}

            <div className="search-box">

              <input
                type="text"
                value={search}
                onChange={(event) =>
                  setSearch(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder="Search asteroid by name or designation"
              />

              <button
                onClick={handleSearch}
                disabled={searching}
              >
                {searching
                  ? "Searching..."
                  : "Search"}
              </button>

            </div>


            {/* ERROR */}

            {error && (
              <div className="error-message">
                {error}
              </div>
            )}


            {/* RESULTS */}

            {results.length > 0 && (
              <div className="results">

                <div className="results-header">
                  <span>
                    Matching asteroids
                  </span>

                  <span>
                    {results.length}
                  </span>
                </div>


                {results.map((asteroid) => (
                  <button
                    key={asteroid.spkid}
                    className="result-item"
                    onClick={() =>
                      handleAnalyze(
                        asteroid.spkid
                      )
                    }
                  >

                    <div>
                      <strong>
                        {asteroid.name}
                      </strong>

                      <span>
                        Designation:{" "}
                        {asteroid.designation}
                      </span>
                    </div>

                    <span className="arrow">
                      →
                    </span>

                  </button>
                ))}

              </div>
            )}


            {loading && (
              <div className="loading">
                Analyzing asteroid...
              </div>
            )}

          </section>

        </main>
      )}


      {/* ================= DASHBOARD PAGE ================= */}

      {page === "dashboard" &&
        selectedAsteroid && (

          <main className="dashboard">

            {/* BACK BUTTON */}

            <button
              className="back-button"
              onClick={handleBack}
            >
              ← Back to Search
            </button>


            {/* TITLE */}

            <section className="dashboard-header">

              <p className="eyebrow">
                ASTRORISK ANALYSIS
              </p>

              <h2>
                {selectedAsteroid.name}
              </h2>

              <div className="asteroid-meta">

                <span>
                  Designation:{" "}
                  {selectedAsteroid.designation}
                </span>

                <span>
                  SPK-ID:{" "}
                  {selectedAsteroid.spkid}
                </span>

              </div>

            </section>


            {/* MAIN RESULT */}

            <section className="result-overview">

              <div className="probability-card">

                <span>
                  ML PHA Probability
                </span>

                <strong>
                  {selectedAsteroid.pha_probability}%
                </strong>

                <p>
                  Estimated probability of
                  PHA classification
                </p>

              </div>


              <div
                className={`classification-card ${
                  selectedAsteroid.classification ===
                  "PHA"
                    ? "pha"
                    : "non-pha"
                }`}
              >

                <span>
                  Classification
                </span>

                <strong>
                  {selectedAsteroid.classification}
                </strong>

                <p>
                  Based on the trained
                  Random Forest model
                </p>

              </div>

            </section>


            {/* ORBITAL PARAMETERS */}

            <section className="dashboard-section">

              <div className="section-title">

                <span className="section-number">
                  01
                </span>

                <div>
                  <h3>
                    Orbital Parameters
                  </h3>

                  <p>
                    Key characteristics of the
                    asteroid's orbit
                  </p>
                </div>

              </div>


              <div className="data-grid">

                <DataCard
                  label="Eccentricity"
                  value={
                    selectedAsteroid
                      .orbital_parameters
                      .eccentricity
                  }
                />

                <DataCard
                  label="Semi-major Axis"
                  value={`${selectedAsteroid
                    .orbital_parameters
                    .semi_major_axis} AU`}
                />

                <DataCard
                  label="Perihelion Distance"
                  value={`${selectedAsteroid
                    .orbital_parameters
                    .perihelion_distance} AU`}
                />

                <DataCard
                  label="Inclination"
                  value={`${selectedAsteroid
                    .orbital_parameters
                    .inclination}°`}
                />

                <DataCard
                  label="Orbital Period"
                  value={`${selectedAsteroid
                    .orbital_parameters
                    .orbital_period} days`}
                />

              </div>

            </section>


            {/* OBSERVATION INFORMATION */}

            <section className="dashboard-section">

              <div className="section-title">

                <span className="section-number">
                  02
                </span>

                <div>
                  <h3>
                    Observation Information
                  </h3>

                  <p>
                    Available observation history
                  </p>
                </div>

              </div>


              <div className="data-grid">

                <DataCard
                  label="Data Arc"
                  value={`${selectedAsteroid
                    .observation_information
                    .data_arc} days`}
                />

                <DataCard
                  label="Observations"
                  value={
                    selectedAsteroid
                      .observation_information
                      .observations
                  }
                />

                <DataCard
                  label="Condition Code"
                  value={
                    selectedAsteroid
                      .observation_information
                      .condition_code
                  }
                />

              </div>

            </section>


            {/* SCIENTIFIC NOTE */}

            <section className="scientific-note">

              <div className="note-icon">
                !
              </div>

              <div>

                <h4>
                  Important Scientific Note
                </h4>

                <p>
                  AstroRisk's probability represents
                  the machine-learning model's estimated
                  probability of PHA classification.
                  It is <strong>not</strong> an asteroid
                  impact probability.
                </p>

              </div>

            </section>


            {/* FOOTER ID */}

            <div className="dashboard-footer">

              <span>
                AstroRisk
              </span>

              <span>
                ML-based asteroid analysis
              </span>

            </div>

          </main>
        )}


      {/* FOOTER */}

      <footer>

        <p>
          AstroRisk • ML-based PHA classification
        </p>

        <p>
          NASA/JPL-derived asteroid data
        </p>

      </footer>

    </div>
  );
}


// REUSABLE DATA CARD

function DataCard({ label, value }) {
  return (
    <div className="data-card">

      <span>
        {label}
      </span>

      <strong>
        {value}
      </strong>

    </div>
  );
}


export default App;