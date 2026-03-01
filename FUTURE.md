# Future Features & Roadmap

## High Impact
- **Web Dashboard** — Live FastAPI page: real-time camera thumbnails, employee status cards, daily hour bars, weekly charts
- **Anomaly Detection** — Flag unusual patterns: low hours, odd clock-in times, statistical threshold alerts
- **Zone-based Tracking** — Define polygon ROIs (lab bench, office, kitchen); track *where* each person spends time
- **Auto-enrollment** — Unknown face appears 5+ times → Discord prompt with snapshot: "Add this person?"
- **Mobile Notifications** — Discord bot or Firebase push: clock-in/out alerts on your phone

## Analytics & Reporting
- **Heatmap Generation** — Spatial heatmaps showing where people spend most time in frame
- **Trend Graphs** — Weekly/monthly working-hour trends as auto-generated charts (matplotlib → Discord/dashboard)
- **CSV/Excel Export** — Exportable attendance data for payroll integration
- **Break Pattern Analysis** — Track break count, average duration, time-of-day patterns per employee
- **Punctuality Score** — Rate employees on late arrivals / early departures over time

## Security & Robustness
- **Anti-spoofing** — Liveness detection (eye blink, head movement) to block photo/screen attacks
- **Tamper Detection** — Alert if camera is covered, moved, or disconnected (sudden darkness / frozen frames)
- **Encrypted Logs** — AES-encrypt JSON attendance logs to prevent tampering
- **Audit Trail** — Log every system action (password entry, config change, manual override) with timestamps

## Smart Features (Advanced)
- **Emotion/Engagement Estimation** — Detect drowsiness or distraction via head pose + eye openness
- **PPE Detection** — Detect required safety gear (lab coat, goggles) by fine-tuning YOLO on PPE classes
- **Occupancy Counting** — Track room headcount over time; useful for capacity limits
- **Collaborative Sessions** — Detect 2+ employees in the same zone; track team collaboration hours
