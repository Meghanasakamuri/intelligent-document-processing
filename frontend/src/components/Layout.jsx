import { NavLink, Outlet } from "react-router-dom";

function Layout() {
  return (
    <div style={{ display: "flex", minHeight: "100vh" }}>

      {/* Sidebar */}
      <aside
        style={{
          width: "240px",
          background: "#172033",
          color: "white",
          padding: "25px 15px",
          boxSizing: "border-box",
        }}
      >
        <h2 style={{ marginBottom: "35px" }}>
          IDP
        </h2>

        <nav style={{ display: "flex", flexDirection: "column", gap: "10px" }}>

          <NavLink to="/" style={linkStyle}>
            Dashboard
          </NavLink>

          <NavLink to="/upload" style={linkStyle}>
            Upload & Process
          </NavLink>

          <NavLink to="/review" style={linkStyle}>
            Document Review
          </NavLink>

          <NavLink to="/history" style={linkStyle}>
            History & Export
          </NavLink>

          <NavLink to="/evaluation" style={linkStyle}>
            Evaluation
          </NavLink>

        </nav>
      </aside>

      {/* Main Content */}
      <main
        style={{
          flex: 1,
          background: "#f5f7fb",
          minHeight: "100vh",
        }}
      >
        <header
          style={{
            background: "white",
            padding: "20px 30px",
            borderBottom: "1px solid #ddd",
          }}
        >
          <h2 style={{ margin: 0 }}>
            Intelligent Document Processing
          </h2>
        </header>

        <section style={{ padding: "30px" }}>
          <Outlet />
        </section>
      </main>

    </div>
  );
}

const linkStyle = ({ isActive }) => ({
  color: "white",
  textDecoration: "none",
  padding: "12px 15px",
  borderRadius: "8px",
  background: isActive ? "#2563eb" : "transparent",
  display: "block",
});

export default Layout;