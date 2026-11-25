import { BrowserRouter, Routes, Route } from "react-router-dom";

import Layout from "./components/Layout";

import Dashboard from "./pages/Dashboard";
import UploadProcess from "./pages/UploadProcess";
import DocumentReview from "./pages/DocumentReview";
import DocumentQA from "./pages/DocumentQA";
import History from "./pages/History";
import Evaluation from "./pages/Evaluation";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<Dashboard />} />
          <Route path="/upload" element={<UploadProcess />} />
          <Route path="/review" element={<DocumentReview />} />
          <Route path="/review/:id" element={<DocumentReview />} />
          <Route path="/qa" element={<DocumentQA />} />
          <Route path="/history" element={<History />} />
          <Route path="/evaluation" element={<Evaluation />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
