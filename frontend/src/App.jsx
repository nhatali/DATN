import { useState } from "react";
import "./App.css";

function App() {

  const [image, setImage] = useState(null);

  const [preview, setPreview] = useState(null);

  const [result, setResult] = useState("");

  const [accuracy, setAccuracy] = useState("");

  const [evaluation, setEvaluation] = useState("");

  const [posture, setPosture] = useState("");

  const [technique, setTechnique] = useState("");

  const [confidenceLevel, setConfidenceLevel] =
    useState("");

  const [topPredictions, setTopPredictions] =
    useState([]);

  // =========================
  // CHỌN ẢNH
  // =========================

  const handleImageChange = (e) => {

    const file = e.target.files[0];

    if (file) {

      setImage(file);

      setPreview(
        URL.createObjectURL(file)
      );

      // RESET RESULT

      setResult("");

      setAccuracy("");

      setEvaluation("");

      setPosture("");

      setTechnique("");

      setConfidenceLevel("");

      setTopPredictions([]);
    }
  };

  // =========================
  // PREDICT
  // =========================

  const handlePredict = async () => {

    if (!image) {

      alert("Vui lòng chọn ảnh");

      return;
    }

    const formData = new FormData();

    formData.append("file", image);

    try {

      const response = await fetch(
        "http://127.0.0.1:5000/predict",
        {
          method: "POST",
          body: formData,
        }
      );

      const data = await response.json();

      console.log(data);

      // RESULT

      setResult(data.prediction);

      setAccuracy(data.accuracy);

      setEvaluation(data.evaluation);

      setPosture(data.posture);

      setTechnique(data.technique);

      setConfidenceLevel(
        data.confidence_level
      );

      setTopPredictions(
        data.top_predictions
      );

    } catch (error) {

      console.log(error);

      alert("Lỗi kết nối backend");
    }
  };

  return (

    <div className="container">

      <div className="main-card">

        <h1>
          Hệ Thống Phân Tích Hiệu Suất Tập Luyện
        </h1>

        <p className="subtitle">
          AI Action Recognition using CNN &
          Federated Learning
        </p>

        {/* ========================= */}
        {/* INPUT */}
        {/* ========================= */}

        <div className="upload-section">

          <input
            type="file"
            accept="image/*"
            onChange={handleImageChange}
          />

          <button onClick={handlePredict}>
            Predict
          </button>

        </div>

        {/* ========================= */}
        {/* IMAGE PREVIEW */}
        {/* ========================= */}

        <div className="preview-box">

          {preview ? (

            <img
              src={preview}
              alt="preview"
            />

          ) : (

            <p>Chưa có ảnh</p>

          )}

        </div>

        {/* ========================= */}
        {/* DASHBOARD */}
        {/* CHỈ HIỆN KHI ĐÃ PREDICT */}
        {/* ========================= */}

        {result && (

          <div className="dashboard">

            {/* ========================= */}
            {/* ACCURACY */}
            {/* ========================= */}

            <div className="card">

              <h2>Độ chính xác</h2>

              <div className="accuracy-circle">

                {accuracy}

              </div>

              <div className="level">

                Mức độ tự tin:

                <span className="high">

                  {" "}
                  {confidenceLevel}

                </span>

              </div>

              <div className="progress-bar">

                <div
                  className="progress"
                  style={{
                    width: accuracy
                  }}
                ></div>

              </div>

            </div>

            {/* ========================= */}
            {/* EVALUATION */}
            {/* ========================= */}

            <div className="card">

              <h2>Đánh giá</h2>

              <div className="evaluation">

                <p>

                  ✔ Tư thế:
                  {" "}
                  {posture}

                </p>

                <p>

                  ✔ Kỹ thuật:
                  {" "}
                  {technique}

                </p>

                <p>

                  ⚠ Đề xuất:
                  {" "}
                  {evaluation}

                </p>

              </div>

            </div>

          </div>

        )}

        {/* ========================= */}
        {/* TOP 5 */}
        {/* ========================= */}

        {result && (

          <div className="prediction-card">

            <h2>Top 5 dự đoán</h2>

            {topPredictions.map(
              (item, index) => (

                <div
                  className="prediction-item"
                  key={index}
                >

                  <div className="prediction-info">

                    <span>

                      {index + 1}.
                      {" "}
                      {item.label}

                    </span>

                    <span>

                      {item.score}%

                    </span>

                  </div>

                  <div className="prediction-bar">

                    <div
                      className="prediction-fill"
                      style={{
                        width:
                          `${item.score}%`
                      }}
                    ></div>

                  </div>

                </div>
              )
            )}

          </div>

        )}

        {/* ========================= */}
        {/* FINAL RESULT */}
        {/* ========================= */}

        {result && (

          <div className="final-result">

            <h2>
              Kết quả nhận dạng
            </h2>

            <p>{result}</p>

          </div>

        )}

      </div>

    </div>
  );
}

export default App;