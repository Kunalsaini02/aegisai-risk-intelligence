"use client";

import { useState } from "react";

export default function Home() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);

  const runSimulation = async (type: "legit" | "fraud") => {
    setLoading(true);

    const legitTransaction = {
      V1: 0.1, V2: -0.2, V3: 0.3, V4: 0.4, V5: 0.1,
      V6: -0.1, V7: 0.2, V8: 0.05, V9: 0.01, V10: -0.02,
      V11: 0.3, V12: -0.4, V13: 0.5, V14: -0.3, V15: 0.2,
      V16: -0.1, V17: 0.01, V18: 0.02, V19: 0.1, V20: 0.05,
      V21: -0.01, V22: 0.02, V23: 0.03, V24: 0.04, V25: 0.05,
      V26: 0.01, V27: 0.02, V28: 0.01,
      Amount: 120,
      Time: 0
    };

    const fraudTransaction = {
  
      V1: -2.3122265423263, V2: 1.95199201064158, V3: -1.60985073229769, V4: 3.9979055875468,
      V5: -0.522187864667764, V6: -1.42654531920595, V7: -2.53738730624579, V8: 1.39165724829804,
      V9: -2.77008927719433, V10: -2.77227214465915, V11: 3.20203320709635, V12: -2.89990738849473,
      V13: -0.595221881324605, V14: -4.28925378244217, V15: 0.389724120274487, V16: -1.14074717980657,
      V17: -2.83005567450437, V18: -0.0168224681808257, V19: 0.416955705037907, V20: 0.126910559061474,
      V21: 0.517232370861764, V22: -0.0350493686052974,V23: -0.465211076182388, V24: 0.320198198514526,
      V25: 0.0445191674731724, V26: 0.177839798284401, V27: 0.261145002567677, V28: -0.143275874698919,
      Amount: 0,
      Time: 406
    };

    const payload = type === "legit" ? legitTransaction : fraudTransaction;

    const response = await fetch("http://127.0.0.1:8000/investigate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });

    const data = await response.json();
    setResult(data);
    setLoading(false);
  };

  return (
    <main className="min-h-screen bg-neutral-950 text-white">
      <div className="max-w-7xl mx-auto px-8 py-16">
        <h1 className="text-4xl font-bold tracking-tight mb-10">
          AegisAI Risk Intelligence
        </h1>

        <div className="grid grid-cols-2 gap-8">

          <div className="bg-neutral-900 p-8 rounded-2xl border border-neutral-800">
            <h2 className="text-xl font-semibold mb-6">
              Transaction Simulation
            </h2>

            <div className="flex gap-4">
              <button
                disabled={loading}
                onClick={() => runSimulation("legit")}
                className="bg-green-600 px-6 py-3 rounded-lg font-medium hover:bg-green-500 transition disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Simulate Legitimate"}
              </button>

              <button
                disabled={loading}
                onClick={() => runSimulation("fraud")}
                className="bg-red-600 px-6 py-3 rounded-lg font-medium hover:bg-red-500 transition disabled:opacity-50"
              >
                {loading ? "Analyzing..." : "Simulate Fraud"}
              </button>
            </div>
          </div>

          <div className="bg-neutral-900 p-8 rounded-2xl border border-neutral-800">
            <h2 className="text-xl font-semibold mb-4">
              Risk Output
            </h2>

            {result ? (
              <div className="space-y-4">
                <div>
                  <p className="text-neutral-400">Fraud Probability</p>
                  <p className="text-2xl font-bold">
                    {(result.fraud_probability * 100).toFixed(4)}%
                  </p>
                </div>

                <div>
                  <p className="text-neutral-400">Risk Level</p>
                  <span
                    className={`inline-block px-3 py-1 rounded-full text-sm font-medium ${
                      result.risk_level === "High"
                        ? "bg-red-500/20 text-red-400"
                        : result.risk_level === "Medium"
                        ? "bg-yellow-500/20 text-yellow-400"
                        : "bg-green-500/20 text-green-400"
                    }`}
                  >
                    {result.risk_level}
                  </span>
                </div>

                <div>
                  <p className="text-neutral-400">Prediction</p>
                  <p>
                    {result.prediction === 1 ? "Fraud" : "Legitimate"}
                  </p>
                </div>

                <div>
                  <p className="text-neutral-400 mb-2">Top Risk Factors</p>
                  <div className="space-y-2">
                    {result.top_risk_factors?.map((factor: any, index: number) => (
                      <div
                        key={index}
                        className="flex justify-between bg-neutral-800 px-3 py-2 rounded-lg"
                      >
                        <span>{factor.feature}</span>
                        <span className="text-neutral-400">
                          {factor.impact.toFixed(4)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-6">
                  <p className="text-neutral-400 mb-2">AI Investigation Report</p>
                  <div className="bg-neutral-800 p-4 rounded-lg text-sm leading-relaxed text-neutral-300 whitespace-pre-wrap">
                    {result.investigation_report}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-neutral-400">
                Choose a simulation to analyze risk.
              </p>
            )}

          </div>

        </div>
      </div>
    </main>
  );
}