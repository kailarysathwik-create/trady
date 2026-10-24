"""TRADE9 Desktop Living AI Overlay Copilot.

A lightweight, transparent, always-on-top floating HUD that sits over any real
trading platform (TradingView, Zerodha Kite, Groww, Binance, ThinkorSwim, etc.),
evaluating live market conditions and providing real-time sniper entry triggers
with zero emojis and institutional precision.
"""

from __future__ import annotations

import ctypes
import json
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk
import urllib.request
from typing import Any


class Trade9OverlayApp:
    """Always-on-top transparent desktop overlay for live trading assistance."""

    def __init__(self, studio_url: str = "http://localhost:8090", default_symbol: str = "SPY") -> None:
        self.studio_url = studio_url.rstrip("/")
        self.symbol = default_symbol.upper()
        self.is_running = True
        self.current_state = "SCANNING"
        self.sound_enabled = True
        self.collapsed = False
        self.drag_start_x = 0
        self.drag_start_y = 0

        self.root = tk.Tk()
        self.root.title("TRADE9 Copilot Overlay")
        self.root.geometry("380x420+40+60")
        self.root.overrideredirect(True)  # Frameless for modern HUD look
        self.root.attributes("-topmost", True)  # Always on top of TradingView/Zerodha
        self.root.attributes("-alpha", 0.92)  # Semi-transparent glass HUD
        self.root.configure(bg="#0B132B")

        self._build_ui()
        self._bind_drag_events()

        # Polling thread for market data & sniper trigger
        self.worker_thread = threading.Thread(target=self._data_polling_loop, daemon=True)
        self.worker_thread.start()

        # Window detection thread
        self.detect_thread = threading.Thread(target=self._window_detection_loop, daemon=True)
        self.detect_thread.start()

    def _build_ui(self) -> None:
        # 1. Header Drag Bar
        self.header_frame = tk.Frame(self.root, bg="#1C2541", height=32)
        self.header_frame.pack(fill=tk.X, side=tk.TOP)

        self.title_lbl = tk.Label(
            self.header_frame,
            text="TRADE9 COPILOT | LIVING AI HUD",
            font=("Segoe UI", 9, "bold"),
            fg="#6FFFE9",
            bg="#1C2541",
            padx=8,
            pady=6,
        )
        self.title_lbl.pack(side=tk.LEFT)

        self.close_btn = tk.Label(
            self.header_frame,
            text="[X]",
            font=("Segoe UI", 9, "bold"),
            fg="#FF5A5F",
            bg="#1C2541",
            padx=8,
            cursor="hand2",
        )
        self.close_btn.pack(side=tk.RIGHT)
        self.close_btn.bind("<Button-1>", lambda e: self.quit())

        self.min_btn = tk.Label(
            self.header_frame,
            text="[-]",
            font=("Segoe UI", 9, "bold"),
            fg="#A5B4FC",
            bg="#1C2541",
            padx=4,
            cursor="hand2",
        )
        self.min_btn.pack(side=tk.RIGHT)
        self.min_btn.bind("<Button-1>", lambda e: self._toggle_collapse())

        # 2. Main Content Container
        self.content_frame = tk.Frame(self.root, bg="#0B132B", padx=12, pady=10)
        self.content_frame.pack(fill=tk.BOTH, expand=True)

        # Active Symbol & Price Bar
        self.price_frame = tk.Frame(self.content_frame, bg="#0B132B")
        self.price_frame.pack(fill=tk.X, pady=(0, 8))

        self.sym_lbl = tk.Label(
            self.price_frame,
            text=self.symbol,
            font=("Segoe UI", 16, "bold"),
            fg="#FFFFFF",
            bg="#0B132B",
        )
        self.sym_lbl.pack(side=tk.LEFT)

        self.price_lbl = tk.Label(
            self.price_frame,
            text="$---.--",
            font=("Consolas", 15, "bold"),
            fg="#6FFFE9",
            bg="#0B132B",
            padx=10,
        )
        self.price_lbl.pack(side=tk.LEFT)

        self.spread_lbl = tk.Label(
            self.price_frame,
            text="SPREAD: $0.00",
            font=("Consolas", 8),
            fg="#94A3B8",
            bg="#0B132B",
        )
        self.spread_lbl.pack(side=tk.RIGHT, pady=4)

        # 3. Dynamic Sniper Trigger Card
        self.trigger_card = tk.Frame(self.content_frame, bg="#1C2541", bd=1, relief=tk.SOLID)
        self.trigger_card.pack(fill=tk.X, pady=(0, 8), padx=2, ipady=6)

        self.state_badge = tk.Label(
            self.trigger_card,
            text="STATE: SCANNING MARKET STRUCTURE",
            font=("Segoe UI", 9, "bold"),
            fg="#CBD5E1",
            bg="#334155",
            padx=8,
            pady=3,
        )
        self.state_badge.pack(fill=tk.X, padx=8, pady=(4, 6))

        self.directive_lbl = tk.Label(
            self.trigger_card,
            text="Analyzing order flow and technical confluence. Stand by for trigger...",
            font=("Segoe UI", 8),
            fg="#E2E8F0",
            bg="#1C2541",
            wraplength=330,
            justify=tk.LEFT,
            padx=8,
        )
        self.directive_lbl.pack(fill=tk.X, padx=4)

        # 4. Tactical Playbook Grid (Entry, Target, Stop)
        self.grid_frame = tk.Frame(self.content_frame, bg="#0B132B")
        self.grid_frame.pack(fill=tk.X, pady=(0, 8))

        # Entry Box
        self.entry_box = tk.Frame(self.grid_frame, bg="#1C2541", padx=6, pady=4)
        self.entry_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(self.entry_box, text="OPTIMAL ENTRY", font=("Segoe UI", 7, "bold"), fg="#94A3B8", bg="#1C2541").pack(anchor=tk.W)
        self.entry_val = tk.Label(self.entry_box, text="---", font=("Consolas", 9, "bold"), fg="#38BDF8", bg="#1C2541")
        self.entry_val.pack(anchor=tk.W)

        # Stop Box
        self.stop_box = tk.Frame(self.grid_frame, bg="#1C2541", padx=6, pady=4)
        self.stop_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(self.stop_box, text="DEFENSIVE STOP", font=("Segoe UI", 7, "bold"), fg="#94A3B8", bg="#1C2541").pack(anchor=tk.W)
        self.stop_val = tk.Label(self.stop_box, text="---", font=("Consolas", 9, "bold"), fg="#EF4444", bg="#1C2541")
        self.stop_val.pack(anchor=tk.W)

        # Target Box
        self.target_box = tk.Frame(self.grid_frame, bg="#1C2541", padx=6, pady=4)
        self.target_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=2)
        tk.Label(self.target_box, text="TARGET 1 (1:2)", font=("Segoe UI", 7, "bold"), fg="#94A3B8", bg="#1C2541").pack(anchor=tk.W)
        self.target_val = tk.Label(self.target_box, text="---", font=("Consolas", 9, "bold"), fg="#10B981", bg="#1C2541")
        self.target_val.pack(anchor=tk.W)

        # 5. Confluence Pill Strip
        self.conf_frame = tk.Frame(self.content_frame, bg="#0B132B")
        self.conf_frame.pack(fill=tk.X, pady=(0, 8))

        self.conf_lbl = tk.Label(
            self.conf_frame,
            text="CONFLUENCE: 0.0% | SETUP: ---",
            font=("Segoe UI", 8, "bold"),
            fg="#A5B4FC",
            bg="#0B132B",
        )
        self.conf_lbl.pack(anchor=tk.W)

        # 6. Quick Symbol Switch Buttons
        self.btn_frame = tk.Frame(self.content_frame, bg="#0B132B")
        self.btn_frame.pack(fill=tk.X, pady=(4, 0))

        presets = ["SPY", "NVDA", "^NSEI", "AAPL", "AMD"]
        for p in presets:
            btn = tk.Label(
                self.btn_frame,
                text=p,
                font=("Consolas", 8, "bold"),
                fg="#FFFFFF",
                bg="#1C2541",
                padx=6,
                pady=2,
                cursor="hand2",
            )
            btn.pack(side=tk.LEFT, padx=2)
            btn.bind("<Button-1>", lambda e, s=p: self.switch_symbol(s))

        self.sound_btn = tk.Label(
            self.btn_frame,
            text="[AUDIO: ON]",
            font=("Consolas", 8, "bold"),
            fg="#10B981",
            bg="#1C2541",
            padx=6,
            pady=2,
            cursor="hand2",
        )
        self.sound_btn.pack(side=tk.RIGHT, padx=2)
        self.sound_btn.bind("<Button-1>", lambda e: self._toggle_sound())

    def _bind_drag_events(self) -> None:
        self.header_frame.bind("<ButtonPress-1>", self._on_drag_start)
        self.header_frame.bind("<B1-Motion>", self._on_drag_motion)
        self.title_lbl.bind("<ButtonPress-1>", self._on_drag_start)
        self.title_lbl.bind("<B1-Motion>", self._on_drag_motion)

    def _on_drag_start(self, event: tk.Event) -> None:
        self.drag_start_x = event.x
        self.drag_start_y = event.y

    def _on_drag_motion(self, event: tk.Event) -> None:
        deltax = event.x - self.drag_start_x
        deltay = event.y - self.drag_start_y
        x = self.root.winfo_x() + deltax
        y = self.root.winfo_y() + deltay
        self.root.geometry(f"+{x}+{y}")

    def _toggle_collapse(self) -> None:
        self.collapsed = not self.collapsed
        if self.collapsed:
            self.content_frame.pack_forget()
            self.root.geometry("380x36")
            self.min_btn.config(text="[+]")
        else:
            self.root.geometry("380x420")
            self.content_frame.pack(fill=tk.BOTH, expand=True)
            self.min_btn.config(text="[-]")

    def _toggle_sound(self) -> None:
        self.sound_enabled = not self.sound_enabled
        if self.sound_enabled:
            self.sound_btn.config(text="[AUDIO: ON]", fg="#10B981")
        else:
            self.sound_btn.config(text="[AUDIO: OFF]", fg="#94A3B8")

    def switch_symbol(self, new_symbol: str) -> None:
        self.symbol = new_symbol.upper().strip()
        self.sym_lbl.config(text=self.symbol)
        self.price_lbl.config(text="$---.--")
        self.state_badge.config(text="STATE: SCANNING", bg="#334155", fg="#CBD5E1")
        self.directive_lbl.config(text=f"Switching analysis to {self.symbol}...")

    def _data_polling_loop(self) -> None:
        """Poll market forecast and live quote every 1.5 seconds."""
        while self.is_running:
            try:
                # 1. Fetch live quote
                quote_url = f"{self.studio_url}/api/market/quote?symbol={self.symbol}"
                with urllib.request.urlopen(quote_url, timeout=3) as resp:
                    quote = json.loads(resp.read().decode("utf-8"))

                # 2. Fetch master forecast with sniper trigger
                forecast_url = f"{self.studio_url}/api/market/forecast?symbol={self.symbol}&horizon=5"
                with urllib.request.urlopen(forecast_url, timeout=3) as resp:
                    data = json.loads(resp.read().decode("utf-8"))

                self.root.after(0, self._update_ui, quote, data)
            except Exception:
                pass
            time.sleep(1.5)

    def _window_detection_loop(self) -> None:
        """Screen reader / window detection co-pilot.
        Scans foreground window title on Windows. If viewing a stock chart,
        auto-switches the active HUD symbol.
        """
        known_symbols = [
            "SPY", "NVDA", "AAPL", "AMD", "PLTR", "MSFT", "GOOGL", "AMZN", "META",
            "NSEI", "NIFTY", "BANKNIFTY", "RELIANCE", "TCS", "INFY", "HDFCBANK"
        ]

        user32 = getattr(ctypes.windll, "user32", None)
        if not user32:
            return

        last_detected = ""
        while self.is_running:
            try:
                hwnd = user32.GetForegroundWindow()
                length = user32.GetWindowTextLengthW(hwnd)
                if length > 0:
                    buff = ctypes.create_unicode_buffer(length + 1)
                    user32.GetWindowTextW(hwnd, buff, length + 1)
                    title = buff.value.upper()

                    # Match symbol in window title (e.g. "TradingView - SPY 590.20")
                    for sym in known_symbols:
                        if sym in title and sym != self.symbol and sym != last_detected:
                            last_detected = sym
                            actual_sym = f"^{sym}" if sym in ("NSEI", "NIFTY") else sym
                            self.root.after(0, self.switch_symbol, actual_sym)
                            break
            except Exception:
                pass
            time.sleep(2.0)

    def _update_ui(self, quote: dict[str, Any], forecast_data: dict[str, Any]) -> None:
        if not self.is_running:
            return

        price = quote.get("last_price") or quote.get("price") or 0.0
        currency_sign = quote.get("currency_symbol") or "$"
        self.price_lbl.config(text=f"{currency_sign}{price:.2f}")

        bid = quote.get("bid", 0.0)
        ask = quote.get("ask", 0.0)
        spread = quote.get("spread", abs(ask - bid))
        self.spread_lbl.config(text=f"SPREAD: {currency_sign}{spread:.2f}")

        mm = forecast_data.get("master_mind", {})
        if not mm:
            return

        sniper = mm.get("sniper_trigger", {})
        state = sniper.get("state", "SCANNING")
        badge = sniper.get("badge", "SCANNING")
        directive = sniper.get("directive", "")

        # Sound Alert Trigger
        if state == "TRIGGERED" and self.current_state != "TRIGGERED" and self.sound_enabled:
            self._beep()

        self.current_state = state

        # State Styling
        if state == "TRIGGERED":
            self.state_badge.config(text=badge, bg="#059669", fg="#FFFFFF")
        elif state == "ARMED":
            self.state_badge.config(text=badge, bg="#D97706", fg="#FFFFFF")
        elif state == "INVALIDATED":
            self.state_badge.config(text=badge, bg="#DC2626", fg="#FFFFFF")
        else:
            self.state_badge.config(text=badge, bg="#334155", fg="#CBD5E1")

        self.directive_lbl.config(text=directive)

        playbook = mm.get("playbook", {})
        entry = playbook.get("entry_zone", [0, 0])
        stop = playbook.get("stop_loss", 0.0)
        target = playbook.get("take_profit_1", 0.0)

        self.entry_val.config(text=f"{currency_sign}{entry[0]:.2f} - {currency_sign}{entry[1]:.2f}")
        self.stop_val.config(text=f"{currency_sign}{stop:.2f}")
        self.target_val.config(text=f"{currency_sign}{target:.2f}")

        score = mm.get("confluence_score", 0.0)
        setup = mm.get("setup_name", "---")
        self.conf_lbl.config(text=f"CONFLUENCE: {score:.1f}% | {setup}")

    def _beep(self) -> None:
        try:
            import winsound
            winsound.Beep(1200, 200)
            winsound.Beep(1600, 250)
        except Exception:
            pass

    def run(self) -> None:
        self.root.mainloop()

    def quit(self) -> None:
        self.is_running = False
        self.root.destroy()


def main() -> None:
    symbol = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    app = Trade9OverlayApp(default_symbol=symbol)
    app.run()


if __name__ == "__main__":
    main()
