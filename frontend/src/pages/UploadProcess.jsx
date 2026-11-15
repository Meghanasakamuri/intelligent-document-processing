import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { uploadDocument } from "../services/api";

function UploadProcess() {
  const navigate = useNavigate();

  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];

    setFile(selectedFile);
    setResult(null);
    setError("");
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a document first.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      /*
       * Get previous document before replacing it.
       */
      const previousDocument =
        localStorage.getItem("lastDocument");

      /*
       * Get existing history.
       */
      let history =
        JSON.parse(
          localStorage.getItem("documentHistory")
        ) || [];

      if (!Array.isArray(history)) {
        history = [];
      }

      /*
       * Preserve previous document.
       */
      if (previousDocument) {
        const oldDocument =
          JSON.parse(previousDocument);

        const alreadyExists =
          history.some(
            (item) =>
              item.filename ===
              oldDocument.filename
          );

        if (!alreadyExists) {
          history.push({
            ...oldDocument,
            verified:
              oldDocument.verified || false,
            processedAt:
              oldDocument.processedAt ||
              new Date().toISOString(),
          });
        }
      }

      /*
       * Send document to backend.
       */
      const data =
        await uploadDocument(file);

      setResult(data);

      /*
       * Create history record.
       */
      const newDocument = {
        ...data,
        verified: false,
        processedAt:
          new Date().toISOString(),
      };

      /*
       * Add newest document first.
       */
      history.unshift(newDocument);

      /*
       * Remove duplicate filenames.
       */
      const uniqueHistory =
        history.filter(
          (item, index, array) =>
            index ===
            array.findIndex(
              (doc) =>
                doc.filename ===
                item.filename
            )
        );

      /*
       * Save history.
       */
      localStorage.setItem(
        "documentHistory",
        JSON.stringify(
          uniqueHistory
        )
      );

      /*
       * Save latest document.
       */
      localStorage.setItem(
        "lastDocument",
        JSON.stringify(
          newDocument
        )
      );

      localStorage.setItem(
        "lastFileName",
        file.name
      );

    } catch (err) {
      console.error(err);

      setError(
        "Unable to process the document. Please make sure the backend is running."
      );
    } finally {
      setLoading(false);
    }
  };

  const openReview = () => {
    navigate("/review");
  };

  return (
    <div>
      <div
        style={{
          marginBottom: "25px",
        }}
      >
        <h1>Upload & Process</h1>

        <p
          style={{
            color: "#64748b",
          }}
        >
          Upload an invoice or document to extract
          important information using OCR.
        </p>
      </div>

      {/* Upload Card */}

      <div
        style={{
          background: "white",
          padding: "30px",
          borderRadius: "12px",
          border:
            "1px solid #e2e8f0",
          marginBottom: "25px",
        }}
      >
        <h2>Select Document</h2>

        <p
          style={{
            color: "#64748b",
          }}
        >
          Supported formats: JPG, JPEG, PNG, PDF
        </p>

        <input
          type="file"
          accept=".jpg,.jpeg,.png,.pdf"
          onChange={handleFileChange}
          style={{
            marginTop: "15px",
            padding: "10px",
          }}
        />

        {file && (
          <div
            style={{
              marginTop: "15px",
              padding: "12px",
              background:
                "#f8fafc",
              borderRadius: "8px",
              border:
                "1px solid #e2e8f0",
            }}
          >
            <p
              style={{
                margin: 0,
                color: "#334155",
              }}
            >
              Selected file:
            </p>

            <strong
              style={{
                display: "block",
                marginTop: "5px",
                overflowWrap:
                  "anywhere",
              }}
            >
              {file.name}
            </strong>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={loading}
          style={{
            marginTop: "20px",
            padding: "13px 25px",
            background: loading
              ? "#94a3b8"
              : "#2563eb",
            color: "white",
            border: "none",
            borderRadius: "8px",
            cursor: loading
              ? "not-allowed"
              : "pointer",
            fontSize: "15px",
            fontWeight: "600",
          }}
        >
          {loading
            ? "Processing..."
            : "Upload & Process"}
        </button>

        {error && (
          <p
            style={{
              marginTop: "15px",
              color: "#dc2626",
            }}
          >
            {error}
          </p>
        )}
      </div>

      {/* Processing Result */}

      {result && (
        <div
          style={{
            background: "white",
            padding: "30px",
            borderRadius: "12px",
            border:
              "1px solid #e2e8f0",
          }}
        >
          <h2>
            Processing Complete
          </h2>

          <p
            style={{
              color: "#16a34a",
              fontWeight: "600",
            }}
          >
            ✓ Document processed successfully
          </p>

          <div
            style={{
              display: "grid",
              gridTemplateColumns:
                "repeat(auto-fit, minmax(200px, 1fr))",
              gap: "15px",
              marginTop: "25px",
            }}
          >
            <ResultCard
              label="Company Name"
              value={
                result
                  .invoice_fields
                  .company_name
              }
            />

            <ResultCard
              label="GST Number"
              value={
                result
                  .invoice_fields
                  .gst_number
              }
            />

            <ResultCard
              label="Invoice Number"
              value={
                result
                  .invoice_fields
                  .invoice_number
              }
            />

            <ResultCard
              label="Invoice Date"
              value={
                result
                  .invoice_fields
                  .invoice_date
              }
            />

            <ResultCard
              label="Total Amount"
              value={
                result
                  .invoice_fields
                  .total_amount
              }
            />

            <ResultCard
              label="Extraction Score"
              value={`${result.invoice_fields.confidence_score}%`}
            />
          </div>

          <button
            onClick={openReview}
            style={{
              marginTop: "25px",
              padding: "13px 25px",
              background: "#16a34a",
              color: "white",
              border: "none",
              borderRadius: "8px",
              cursor: "pointer",
              fontSize: "15px",
              fontWeight: "600",
            }}
          >
            Open Document Review
          </button>
        </div>
      )}
    </div>
  );
}

function ResultCard({
  label,
  value,
}) {
  return (
    <div
      style={{
        padding: "18px",
        background: "#f8fafc",
        borderRadius: "10px",
        border:
          "1px solid #e2e8f0",
      }}
    >
      <p
        style={{
          margin:
            "0 0 8px",
          color: "#64748b",
          fontSize: "13px",
        }}
      >
        {label}
      </p>

      <strong
        style={{
          color: "#334155",
          overflowWrap:
            "anywhere",
        }}
      >
        {value ||
          "Not found"}
      </strong>
    </div>
  );
}

export default UploadProcess;