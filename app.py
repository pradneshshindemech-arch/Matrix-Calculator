"""
Matrix Pro — Python Backend
NumPy math engine exposed to the React frontend via PyWebView's JS API bridge.
"""

import json
import os
import numpy as np
import webview


class MatrixAPI:
    """All matrix operations exposed to the frontend via window.pywebview.api"""

    # ─── Helpers ────────────────────────────────────────────────────────────

    def _parse(self, matrix_json: str) -> np.ndarray:
        data = json.loads(matrix_json)
        return np.array(data, dtype=float)

    def _to_list(self, arr: np.ndarray) -> list:
        """Round near-zero values to 0 for clean display."""
        rounded = np.where(np.abs(arr) < 1e-10, 0.0, arr)
        return rounded.tolist()

    def _result(self, **kwargs) -> str:
        return json.dumps({"ok": True, **kwargs})

    def _error(self, msg: str) -> str:
        return json.dumps({"ok": False, "error": msg})

    # ─── Single-matrix ops ──────────────────────────────────────────────────

    def determinant(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            if A.shape[0] != A.shape[1]:
                return self._error("Matrix must be square")
            det = float(np.linalg.det(A))
            det = 0.0 if abs(det) < 1e-10 else round(det, 6)
            return self._result(value=det)
        except Exception as e:
            return self._error(str(e))

    def inverse(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            if A.shape[0] != A.shape[1]:
                return self._error("Matrix must be square")
            det = np.linalg.det(A)
            if abs(det) < 1e-10:
                return self._error("SINGULAR: det = 0, inverse does not exist")
            inv = np.linalg.inv(A)
            return self._result(matrix=self._to_list(inv))
        except Exception as e:
            return self._error(str(e))

    def transpose(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            return self._result(matrix=self._to_list(A.T))
        except Exception as e:
            return self._error(str(e))

    def rank(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            return self._result(value=int(np.linalg.matrix_rank(A)))
        except Exception as e:
            return self._error(str(e))

    def trace(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            if A.shape[0] != A.shape[1]:
                return self._error("Matrix must be square for trace")
            return self._result(value=float(round(np.trace(A), 6)))
        except Exception as e:
            return self._error(str(e))

    def power(self, matrix_json: str, n: int) -> str:
        try:
            A = self._parse(matrix_json)
            if A.shape[0] != A.shape[1]:
                return self._error("Matrix must be square")
            n = int(n)
            if n == 0:
                result = np.eye(A.shape[0])
            elif n < 0:
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    return self._error("Singular matrix has no negative power")
                result = np.linalg.matrix_power(A, n)
            else:
                result = np.linalg.matrix_power(A, n)
            return self._result(matrix=self._to_list(result))
        except Exception as e:
            return self._error(str(e))

    def eigenvalues(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json)
            if A.shape[0] != A.shape[1]:
                return self._error("Matrix must be square")
            vals = np.linalg.eigvals(A)
            formatted = [
                {"real": round(float(v.real), 4), "imag": round(float(v.imag), 4)}
                for v in vals
            ]
            return self._result(eigenvalues=formatted)
        except Exception as e:
            return self._error(str(e))

    # ─── Two-matrix ops ─────────────────────────────────────────────────────

    def add(self, a_json: str, b_json: str) -> str:
        try:
            A, B = self._parse(a_json), self._parse(b_json)
            if A.shape != B.shape:
                return self._error(f"Shape mismatch: {A.shape} vs {B.shape}")
            return self._result(matrix=self._to_list(A + B))
        except Exception as e:
            return self._error(str(e))

    def subtract(self, a_json: str, b_json: str) -> str:
        try:
            A, B = self._parse(a_json), self._parse(b_json)
            if A.shape != B.shape:
                return self._error(f"Shape mismatch: {A.shape} vs {B.shape}")
            return self._result(matrix=self._to_list(A - B))
        except Exception as e:
            return self._error(str(e))

    def multiply(self, a_json: str, b_json: str) -> str:
        try:
            A, B = self._parse(a_json), self._parse(b_json)
            if A.shape[1] != B.shape[0]:
                return self._error(
                    f"Incompatible shapes: {A.shape} × {B.shape}"
                )
            return self._result(matrix=self._to_list(A @ B))
        except Exception as e:
            return self._error(str(e))

    # ─── Row reduction (Gauss-Jordan) with steps ────────────────────────────

    def row_reduce(self, matrix_json: str) -> str:
        try:
            A = self._parse(matrix_json).astype(float)
            rows, cols = A.shape
            steps = []
            pivot_row = 0

            for col in range(cols):
                if pivot_row >= rows:
                    break
                # Find pivot
                max_row = pivot_row + np.argmax(np.abs(A[pivot_row:, col]))
                if abs(A[max_row, col]) < 1e-10:
                    continue
                # Swap
                if max_row != pivot_row:
                    A[[pivot_row, max_row]] = A[[max_row, pivot_row]]
                    steps.append({
                        "op": "swap",
                        "rows": [pivot_row, max_row],
                        "matrix": self._to_list(A.copy()),
                        "desc": f"R{pivot_row+1} ↔ R{max_row+1}"
                    })
                # Scale
                scale = A[pivot_row, col]
                A[pivot_row] /= scale
                steps.append({
                    "op": "scale",
                    "row": pivot_row,
                    "matrix": self._to_list(A.copy()),
                    "desc": f"R{pivot_row+1} ÷ {round(scale,3)}"
                })
                # Eliminate
                for r in range(rows):
                    if r != pivot_row and abs(A[r, col]) > 1e-10:
                        factor = A[r, col]
                        A[r] -= factor * A[pivot_row]
                        steps.append({
                            "op": "eliminate",
                            "rows": [r, pivot_row],
                            "matrix": self._to_list(A.copy()),
                            "desc": f"R{r+1} − {round(factor,3)}·R{pivot_row+1}"
                        })
                pivot_row += 1

            return self._result(steps=steps, final=self._to_list(A))
        except Exception as e:
            return self._error(str(e))


# ─── PyWebView entrypoint ────────────────────────────────────────────────────

if __name__ == "__main__":
    api = MatrixAPI()
    ui_path = os.path.join(os.path.dirname(__file__), "ui", "dist", "index.html")

    if not os.path.exists(ui_path):
        # Dev fallback: serve from public folder
        ui_path = os.path.join(os.path.dirname(__file__), "ui", "public", "index.html")

    window = webview.create_window(
        title="Matrix Pro",
        url=f"file://{ui_path}",
        js_api=api,
        width=1400,
        height=900,
        min_size=(900, 600),
        background_color="#0a0a1a",
        text_select=False,
    )

    webview.start(debug=False)