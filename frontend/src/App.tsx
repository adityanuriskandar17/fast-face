import { useState } from "react";
import EnrollForm from "./EnrollForm";
import LiveFace from "./LiveFace";

type View = "live" | "enroll";

function App() {
  const [view, setView] = useState<View>("live");

  return (
    <div className="app">
      <header className="app__header">
        <div>
          <span className="eyebrow">Check-in wajah</span>
          <h1 className="wordmark">
            FAST<span className="wordmark__accent">FACE</span>
          </h1>
        </div>
        <nav className="menu">
          <button className={`menu__item ${view === "live" ? "menu__item--active" : ""}`} onClick={() => setView("live")}>
            Live
          </button>
          <button
            className={`menu__item ${view === "enroll" ? "menu__item--active" : ""}`}
            onClick={() => setView("enroll")}
          >
            Daftar Wajah
          </button>
        </nav>
        <span className="gate-tag">Gate 01</span>
      </header>

      <main className={`app__main ${view === "live" ? "app__main--single" : ""}`}>
        {view === "live" ? (
          <section>
            <h2 className="panel__title">Kamera langsung</h2>
            <LiveFace />
          </section>
        ) : (
          <EnrollForm />
        )}
      </main>

      <footer className="app__footer">
        <span className="legend-item">
          <span className="legend-swatch legend-swatch--clear" aria-hidden />
          Kotak hijau — anggota terdaftar
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch--stop" aria-hidden />
          Kotak merah — tidak dikenal
        </span>
        <span className="legend-item">
          <span className="legend-swatch legend-swatch--spoof" aria-hidden />
          Kotak ungu — foto/layar, bukan wajah asli
        </span>
      </footer>
    </div>
  );
}

export default App;
