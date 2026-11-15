import { useEffect, useState } from "react";

function DocumentReview() {
  const [document, setDocument] = useState(null);
  const [fields, setFields] = useState({});
  const [verified, setVerified] = useState(false);

  useEffect(() => {
    const savedDocument = localStorage.getItem("lastDocument");

    if (savedDocument) {
      const data = JSON.parse(savedDocument);

      setDocument(data);
      setFields(data.invoice_fields || {});
      setVerified(data.verified || false);
    }
  }, []);

  const updateField = (field, value) => {
    setFields((previousFields) => ({
      ...previousFields,
      [field]: value,
    }));

    setVerified(false);
  };

  const saveVerification = () => {
    const updatedDocument = {
      ...document,
      invoice_fields: fields,
      verified: true,
    };

    // Update latest document
    localStorage.setItem(
      "lastDocument",
      JSON.stringify(updatedDocument)
    );

    // Update document history
    const history =
      JSON.parse(
        localStorage.getItem("documentHistory")
      ) || [];

    const updatedHistory = history.map((item) => {
      if (
        item.filename === updatedDocument.filename
      ) {
        return {
          ...item,
          invoice_fields: fields,
          verified: true,
        };
      }

      return item;
    });

    localStorage.setItem(
      "documentHistory",
      JSON.stringify(updatedHistory)
    );

    setDocument(updatedDocument);
    setVerified(true);
  };

  if (!document) {
    return (
      <div>
        <h1>Document Review</h1>

        <div
          style={{
            background: "white",
            padding: "35px",
            borderRadius: "12px",
            border: "1px solid #e2e8f0",
          }}
        >
          <h2>No document available</h2>

          <p style={{ color: "#64748b" }}>
            Upload and process a document before opening the review page.
          </p>

          <a
            href="/upload"
            style={{
              display: "inline-block",
              marginTop: "15px",
              padding: "12px 20px",
              background: "#2563eb",
              color: "white",
              textDecoration: "none",
              borderRadius: "8px",
            }}
          >
            Upload Document
          </a>
        </div>
      </div>
    );
  }

  const confidence =
    document.invoice_fields?.confidence_score || 0;

  const fileName =
    localStorage.getItem("lastFileName") ||
    document.filename ||
    "Uploaded document";

  return (
    <div>
      <div style={{ marginBottom: "25px" }}>
        <h1>Document Review</h1>

        <p style={{ color: "#64748b" }}>
          Review, correct and verify the information extracted from the document.
        </p>
      </div>

      {/* Verification Status */}

      <div
        style={{
          padding: "18px 20px",
          borderRadius: "10px",
          marginBottom: "25px",
          background: verified
            ? "#ecfdf5"
            : "#fff7ed",
          border: verified
            ? "1px solid #a7f3d0"
            : "1px solid #fed7aa",
        }}
      >
        <strong>
          {verified
            ? "✓ Document Verified"
            : "⚠ Verification Required"}
        </strong>

        <p
          style={{
            margin: "6px 0 0",
            color: "#64748b",
          }}
        >
          {verified
            ? "The extracted information has been reviewed and verified."
            : "Please review the extracted fields before verification."}
        </p>
      </div>

      {/* Document Information */}

      <div
        style={{
          background: "white",
          padding: "25px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
          marginBottom: "25px",
        }}
      >
        <h2>Document Information</h2>

        <div
          style={{
            display: "grid",
            gridTemplateColumns:
              "repeat(auto-fit, minmax(220px, 1fr))",
            gap: "25px",
            marginTop: "20px",
          }}
        >
          <InfoItem
            label="File Name"
            value={fileName}
            wrap
          />

          <InfoItem
            label="Document Type"
            value="Invoice"
          />

          <InfoItem
            label="Extraction Score"
            value={`${confidence}%`}
          />

          <InfoItem
            label="Processing Status"
            value={
              verified
                ? "Verified"
                : "Needs Review"
            }
          />
        </div>
      </div>

      {/* Extracted Information */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border: "1px solid #e2e8f0",
        }}
      >
        <h2>Extracted Information</h2>

        <p style={{ color: "#64748b" }}>
          Correct any OCR errors before confirming the document.
        </p>

        <div style={{ marginTop: "25px" }}>
          <Field
            label="Company Name"
            value={fields.company_name}
            onChange={(value) =>
              updateField("company_name", value)
            }
          />

          <Field
            label="GST Number"
            value={fields.gst_number}
            onChange={(value) =>
              updateField("gst_number", value)
            }
          />

          <Field
            label="Invoice Number"
            value={fields.invoice_number}
            onChange={(value) =>
              updateField("invoice_number", value)
            }
          />

          <Field
            label="Invoice Date"
            value={fields.invoice_date}
            onChange={(value) =>
              updateField("invoice_date", value)
            }
          />

          <Field
            label="Total Amount"
            value={fields.total_amount}
            onChange={(value) =>
              updateField("total_amount", value)
            }
          />
        </div>

        <div
          style={{
            marginTop: "30px",
            paddingTop: "25px",
            borderTop: "1px solid #e2e8f0",
          }}
        >
          <button
            onClick={saveVerification}
            disabled={verified}
            style={{
              padding: "13px 25px",
              background: verified
                ? "#94a3b8"
                : "#16a34a",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: verified
                ? "not-allowed"
                : "pointer",
              fontSize: "15px",
              fontWeight: "600",
            }}
          >
            {verified
              ? "✓ Document Verified"
              : "✓ Save & Verify Document"}
          </button>
        </div>
      </div>
    </div>
  );
}

function Field({ label, value, onChange }) {
  return (
    <div style={{ marginBottom: "22px" }}>
      <label
        style={{
          display: "block",
          marginBottom: "8px",
          fontWeight: "600",
          color: "#334155",
        }}
      >
        {label}
      </label>

      <input
        type="text"
        value={value || ""}
        onChange={(e) =>
          onChange(e.target.value)
        }
        style={{
          width: "100%",
          boxSizing: "border-box",
          padding: "12px",
          border: "1px solid #cbd5e1",
          borderRadius: "8px",
          fontSize: "15px",
          outline: "none",
        }}
      />
    </div>
  );
}

function InfoItem({
  label,
  value,
  wrap = false,
}) {
  return (
    <div style={{ minWidth: 0 }}>
      <p
        style={{
          margin: "0 0 6px",
          color: "#64748b",
          fontSize: "13px",
        }}
      >
        {label}
      </p>

      <strong
        style={{
          color: "#334155",
          display: "block",
          overflowWrap: wrap
            ? "anywhere"
            : "normal",
          wordBreak: wrap
            ? "break-word"
            : "normal",
        }}
      >
        {value}
      </strong>
    </div>
  );
}

export default DocumentReview;