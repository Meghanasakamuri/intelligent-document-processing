import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

function History() {
  const [documents, setDocuments] = useState([]);
  const [search, setSearch] = useState("");
  const [filter, setFilter] = useState("All");

  useEffect(() => {
    loadHistory();
  }, []);

  const loadHistory = () => {
    const history =
      JSON.parse(
        localStorage.getItem("documentHistory")
      ) || [];

    setDocuments(history);
  };

  const exportDocument = (document) => {
    const data = JSON.stringify(
      document,
      null,
      2
    );

    const blob = new Blob(
      [data],
      { type: "application/json" }
    );

    const url =
      URL.createObjectURL(blob);

    const link =
      window.document.createElement("a");

    link.href = url;
    link.download =
      document.filename
        ? `${document.filename}.json`
        : "processed-document.json";

    link.click();

    URL.revokeObjectURL(url);
  };

  const filteredDocuments =
    documents.filter((document) => {
      const fields =
        document.invoice_fields || {};

      const searchText =
        search.toLowerCase();

      const matchesSearch =
        document.filename
          ?.toLowerCase()
          .includes(searchText) ||
        fields.company_name
          ?.toLowerCase()
          .includes(searchText) ||
        fields.invoice_number
          ?.toLowerCase()
          .includes(searchText);

      const matchesFilter =
        filter === "All" ||
        (filter === "Verified" &&
          document.verified === true) ||
        (filter === "Needs Review" &&
          document.verified !== true);

      return matchesSearch && matchesFilter;
    });

  return (
    <div>
      {/* Header */}

      <div style={{ marginBottom: "25px" }}>
        <h1>History & Export</h1>

        <p style={{ color: "#64748b" }}>
          View, search and export previously processed documents.
        </p>
      </div>

      {/* Search and Filter */}

      <div
        style={{
          background: "white",
          padding: "20px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          marginBottom: "25px",
          display: "flex",
          gap: "15px",
          flexWrap: "wrap",
        }}
      >
        <input
          type="text"
          placeholder="Search by filename, company or invoice number..."
          value={search}
          onChange={(e) =>
            setSearch(e.target.value)
          }
          style={{
            flex: 1,
            minWidth: "250px",
            padding: "12px 15px",
            border: "1px solid #cbd5e1",
            borderRadius: "8px",
            fontSize: "14px",
            outline: "none",
          }}
        />

        <select
          value={filter}
          onChange={(e) =>
            setFilter(e.target.value)
          }
          style={{
            padding: "12px 15px",
            border: "1px solid #cbd5e1",
            borderRadius: "8px",
            fontSize: "14px",
            background: "white",
          }}
        >
          <option value="All">
            All Documents
          </option>

          <option value="Verified">
            Verified
          </option>

          <option value="Needs Review">
            Needs Review
          </option>
        </select>
      </div>

      {/* Document Count */}

      <div
        style={{
          marginBottom: "15px",
          color: "#64748b",
          fontSize: "14px",
        }}
      >
        Showing{" "}
        <strong>
          {filteredDocuments.length}
        </strong>{" "}
        of{" "}
        <strong>
          {documents.length}
        </strong>{" "}
        documents
      </div>

      {/* Empty State */}

      {filteredDocuments.length === 0 ? (
        <div
          style={{
            background: "white",
            padding: "45px",
            borderRadius: "12px",
            border: "1px solid #e2e8f0",
            textAlign: "center",
          }}
        >
          <h2>
            {documents.length === 0
              ? "No Documents Found"
              : "No Matching Documents"}
          </h2>

          <p
            style={{
              color: "#64748b",
            }}
          >
            {documents.length === 0
              ? "Process a document to see it in history."
              : "Try a different search or filter."}
          </p>
        </div>
      ) : (
        <div>
          {filteredDocuments.map(
            (document, index) => {
              const fields =
                document.invoice_fields ||
                {};

              const confidence =
                fields.confidence_score ||
                0;

              return (
                <div
                  key={`${document.filename}-${index}`}
                  style={{
                    background: "white",
                    padding: "25px",
                    borderRadius: "12px",
                    border: "1px solid #e2e8f0",
                    marginBottom: "20px",
                  }}
                >
                  {/* Document Header */}

                  <div
                    style={{
                      display: "flex",
                      justifyContent:
                        "space-between",
                      alignItems: "flex-start",
                      gap: "20px",
                      flexWrap: "wrap",
                      marginBottom: "20px",
                    }}
                  >
                    <div
                      style={{
                        minWidth: 0,
                        flex: 1,
                      }}
                    >
                      <h2
                        style={{
                          margin: 0,
                          overflowWrap:
                            "anywhere",
                        }}
                      >
                        {document.filename ||
                          "Processed Document"}
                      </h2>

                      <p
                        style={{
                          margin:
                            "7px 0 0",
                          color:
                            "#64748b",
                        }}
                      >
                        Invoice
                      </p>
                    </div>

                    <span
                      style={{
                        padding:
                          "7px 13px",
                        borderRadius:
                          "20px",
                        background:
                          document.verified
                            ? "#dcfce7"
                            : "#fff7ed",
                        color:
                          document.verified
                            ? "#166534"
                            : "#c2410c",
                        fontSize:
                          "13px",
                        fontWeight:
                          "600",
                        whiteSpace:
                          "nowrap",
                      }}
                    >
                      {document.verified
                        ? "Verified"
                        : "Needs Review"}
                    </span>
                  </div>

                  {/* Extracted Fields */}

                  <div
                    style={{
                      display:
                        "grid",
                      gridTemplateColumns:
                        "repeat(auto-fit, minmax(180px, 1fr))",
                      gap: "20px",
                      padding:
                        "20px",
                      background:
                        "#f8fafc",
                      borderRadius:
                        "10px",
                    }}
                  >
                    <InfoItem
                      label="Company"
                      value={
                        fields.company_name ||
                        "Not found"
                      }
                    />

                    <InfoItem
                      label="GST Number"
                      value={
                        fields.gst_number ||
                        "Not found"
                      }
                    />

                    <InfoItem
                      label="Invoice Number"
                      value={
                        fields.invoice_number ||
                        "Not found"
                      }
                    />

                    <InfoItem
                      label="Invoice Date"
                      value={
                        fields.invoice_date ||
                        "Not found"
                      }
                    />

                    <InfoItem
                      label="Total Amount"
                      value={
                        fields.total_amount ||
                        "Not found"
                      }
                    />

                    <InfoItem
                      label="Extraction Score"
                      value={`${confidence}%`}
                    />
                  </div>

                  {/* Actions */}

                  <div
                    style={{
                      marginTop:
                        "20px",
                      display:
                        "flex",
                      justifyContent:
                        "flex-end",
                      gap: "10px",
                      flexWrap:
                        "wrap",
                    }}
                  >
                    <Link
                      to="/review"
                      style={{
                        padding:
                          "10px 18px",
                        background:
                          "#e2e8f0",
                        color:
                          "#1e293b",
                        textDecoration:
                          "none",
                        borderRadius:
                          "8px",
                        fontSize:
                          "14px",
                        fontWeight:
                          "600",
                      }}
                    >
                      View Review
                    </Link>

                    <button
                      onClick={() =>
                        exportDocument(
                          document
                        )
                      }
                      style={{
                        padding:
                          "10px 18px",
                        background:
                          "#2563eb",
                        color:
                          "white",
                        border:
                          "none",
                        borderRadius:
                          "8px",
                        cursor:
                          "pointer",
                        fontSize:
                          "14px",
                        fontWeight:
                          "600",
                      }}
                    >
                      Export JSON
                    </button>
                  </div>
                </div>
              );
            }
          )}
        </div>
      )}
    </div>
  );
}

function InfoItem({
  label,
  value,
}) {
  return (
    <div
      style={{
        minWidth: 0,
      }}
    >
      <p
        style={{
          margin:
            "0 0 6px",
          color:
            "#64748b",
          fontSize:
            "13px",
        }}
      >
        {label}
      </p>

      <strong
        style={{
          color:
            "#334155",
          overflowWrap:
            "anywhere",
        }}
      >
        {value}
      </strong>
    </div>
  );
}

export default History;