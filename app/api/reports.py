"""
Reports API Router
Generates structured Excel workbooks using openpyxl and produces aggregated analytics.
Adheres to PAUD LABORATORIUM UM KOTA BLITAR institutional branding.
"""

import io
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends, HTTPException, status, Query, Response
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from app.models.auth import TokenData
from app.models.user import UserRole
from app.security.auth import require_admin
from app.core.firebase import get_firestore_client
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/reports", tags=["Reports"])

WIB = timezone(timedelta(hours=7), name="WIB")

def _get_current_wib() -> datetime:
    return datetime.now(WIB)

@router.get("/attendance/excel")
async def export_attendance_excel(
    startDate: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    endDate: Optional[str] = Query(None, description="Format: YYYY-MM-DD"),
    department: Optional[str] = Query(None),
    admin_user: TokenData = Depends(require_admin)
):
    """
    Exports attendance records to a styled Microsoft Excel workbook (.xlsx).
    Includes a detailed attendance ledger and an employee summary recap sheet.
    """
    db = get_firestore_client()
    if db is None:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database service unavailable.")

    # 1. Fetch attendance records
    query = db.collection("attendance")
    if department:
        query = query.where("department", "==", department)

    docs = query.limit(1000).stream()
    records = []
    for doc in docs:
        data = doc.to_dict()
        w_date = data.get("workDate", "")
        if startDate and w_date < startDate:
            continue
        if endDate and w_date > endDate:
            continue
        records.append({**data, "attendanceId": doc.id})

    # Sort records chronologically descending
    records.sort(key=lambda r: (r.get("workDate", ""), r.get("employeeName", "")), reverse=True)

    # 2. Build OpenPyXL Workbook
    wb = openpyxl.Workbook()

    # --- Styles Definition ---
    navy_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    light_blue_fill = PatternFill(start_color="3B82F6", end_color="3B82F6", fill_type="solid")
    zebra_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    green_fill = PatternFill(start_color="DCFCE7", end_color="DCFCE7", fill_type="solid")
    amber_fill = PatternFill(start_color="FEF3C7", end_color="FEF3C7", fill_type="solid")
    red_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")

    title_font = Font(name="Calibri", size=14, bold=True, color="FFFFFF")
    subtitle_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    meta_font = Font(name="Calibri", size=10, italic=True, color="334155")
    header_font = Font(name="Calibri", size=11, bold=True, color="FFFFFF")
    data_font = Font(name="Calibri", size=10)
    bold_data_font = Font(name="Calibri", size=10, bold=True)
    green_font = Font(name="Calibri", size=10, bold=True, color="166534")
    amber_font = Font(name="Calibri", size=10, bold=True, color="92400E")
    red_font = Font(name="Calibri", size=10, bold=True, color="991B1B")

    thin_border_side = Side(style="thin", color="CBD5E1")
    cell_border = Border(left=thin_border_side, right=thin_border_side, top=thin_border_side, bottom=thin_border_side)
    thick_bottom = Border(bottom=Side(style="medium", color="1E3A8A"))

    center_align = Alignment(horizontal="center", vertical="center")
    left_align = Alignment(horizontal="left", vertical="center")
    right_align = Alignment(horizontal="right", vertical="center")

    # ================= SHEET 1: DETAIL PRESENSI =================
    ws1 = wb.active
    ws1.title = "Detail Presensi"
    ws1.views.sheetView[0].showGridLines = True

    # Title Banner
    ws1.merge_cells("A1:M1")
    title_cell = ws1["A1"]
    title_cell.value = f"PAUD LABORATORIUM UNIVERSITAS NEGERI MALANG KOTA BLITAR"
    title_cell.font = title_font
    title_cell.fill = navy_fill
    title_cell.alignment = center_align
    ws1.row_dimensions[1].height = 28

    ws1.merge_cells("A2:M2")
    sub_cell = ws1["A2"]
    sub_cell.value = "LAPORAN DETAIL PRESENSI KEHADIRAN PEGAWAI & GURU — CHECKHADIR.ID"
    sub_cell.font = subtitle_font
    sub_cell.fill = light_blue_fill
    sub_cell.alignment = center_align
    ws1.row_dimensions[2].height = 22

    now_str = _get_current_wib().strftime("%d-%m-%Y %H:%M WIB")
    period_str = f"Periode: {startDate or 'Semua'} s/d {endDate or 'Hari Ini'}  |  Unit: {department or 'Semua Departemen'}  |  Diunduh: {now_str}"
    ws1["A3"] = period_str
    ws1["A3"].font = meta_font
    ws1.row_dimensions[3].height = 20

    # Headers
    headers = [
        "No",
        "NIY / ID Pegawai",
        "Nama Pegawai",
        "Unit / Departemen",
        "Tanggal",
        "Jam Masuk",
        "Status Masuk",
        "Keterlambatan (Menit)",
        "Jam Pulang",
        "Status Pulang",
        "Verifikasi Wajah",
        "Jarak GPS",
        "Catatan"
    ]

    for col_idx, header in enumerate(headers, 1):
        cell = ws1.cell(row=5, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = navy_fill
        cell.alignment = center_align
        cell.border = cell_border
    ws1.row_dimensions[5].height = 26

    # Data Rows
    current_row = 6
    for idx, r in enumerate(records, 1):
        # Format timestamps
        in_time_val = r.get("checkInAt")
        in_str = "-"
        if in_time_val:
            if hasattr(in_time_val, "strftime"):
                in_str = in_time_val.strftime("%H:%M WIB")
            elif isinstance(in_time_val, str):
                in_str = in_time_val[:16]

        out_time_val = r.get("checkOutAt")
        out_str = "-"
        if out_time_val:
            if hasattr(out_time_val, "strftime"):
                out_str = out_time_val.strftime("%H:%M WIB")
            elif isinstance(out_time_val, str):
                out_str = out_time_val[:16]

        in_status = r.get("checkInStatus")
        status_display = "Tepat Waktu" if in_status == "on_time" else ("Terlambat" if in_status == "late" else "-")
        out_status = r.get("checkOutStatus")
        out_display = "Normal" if out_status == "normal" else ("Pulang Cepat" if out_status == "early_departure" else "-")

        dist_val = "-"
        loc = r.get("checkInLocation")
        if isinstance(loc, dict) and loc.get("distanceMeters") is not None:
            dist_val = f"{loc['distanceMeters']} m"

        face_verified = "SFace Cocok" if r.get("faceVerified") else "-"

        row_data = [
            idx,
            r.get("employeeId", "-"),
            r.get("employeeName", "Pegawai"),
            r.get("department", "-"),
            r.get("workDate", "-"),
            in_str,
            status_display,
            r.get("lateMinutes", 0),
            out_str,
            out_display,
            face_verified,
            dist_val,
            r.get("notes") or ""
        ]

        row_fill = zebra_fill if idx % 2 == 1 else None

        for col_idx, val in enumerate(row_data, 1):
            cell = ws1.cell(row=current_row, column=col_idx, value=val)
            cell.font = data_font
            cell.border = cell_border
            if row_fill:
                cell.fill = row_fill

            # Alignments
            if col_idx in [1, 5, 6, 7, 8, 9, 10, 11, 12]:
                cell.alignment = center_align
            elif col_idx in [2, 3, 4, 13]:
                cell.alignment = left_align

            # Conditional Highlights
            if col_idx == 7:  # Status Masuk
                if status_display == "Tepat Waktu":
                    cell.fill = green_fill
                    cell.font = green_font
                elif status_display == "Terlambat":
                    cell.fill = amber_fill
                    cell.font = amber_font

            if col_idx == 10 and out_display == "Pulang Cepat":
                cell.fill = red_fill
                cell.font = red_font

        ws1.row_dimensions[current_row].height = 20
        current_row += 1

    # Auto-fit columns Sheet 1
    for col in ws1.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            # Skip title row from length computation
            if cell.row in [1, 2, 3]:
                continue
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws1.column_dimensions[col_letter].width = max(max_len + 4, 11)

    # ================= SHEET 2: REKAPITULASI PEGAWAI =================
    ws2 = wb.create_sheet(title="Rekapitulasi Pegawai")
    ws2.views.sheetView[0].showGridLines = True

    # Title
    ws2.merge_cells("A1:H1")
    t2 = ws2["A1"]
    t2.value = "REKAPITULASI TINGKAT KEHADIRAN PEGAWAI & GURU — PAUD LAB UM KOTA BLITAR"
    t2.font = title_font
    t2.fill = navy_fill
    t2.alignment = center_align
    ws2.row_dimensions[1].height = 28

    ws2["A2"] = period_str
    ws2["A2"].font = meta_font
    ws2.row_dimensions[2].height = 20

    # Headers Sheet 2
    sum_headers = [
        "No",
        "NIY / ID Pegawai",
        "Nama Pegawai",
        "Unit / Departemen",
        "Total Hadir",
        "Tepat Waktu",
        "Terlambat",
        "Total Menit Terlambat"
    ]
    for col_idx, h in enumerate(sum_headers, 1):
        cell = ws2.cell(row=4, column=col_idx, value=h)
        cell.font = header_font
        cell.fill = navy_fill
        cell.alignment = center_align
        cell.border = cell_border
    ws2.row_dimensions[4].height = 26

    # Aggregate by employee
    emp_map = {}
    for r in records:
        emp_id = r.get("employeeId") or r.get("uid")
        if not emp_id:
            continue
        if emp_id not in emp_map:
            emp_map[emp_id] = {
                "employeeId": r.get("employeeId", "-"),
                "name": r.get("employeeName", "Pegawai"),
                "department": r.get("department", "-"),
                "totalAttended": 0,
                "onTime": 0,
                "late": 0,
                "totalLateMin": 0
            }
        emp_map[emp_id]["totalAttended"] += 1
        if r.get("checkInStatus") == "on_time":
            emp_map[emp_id]["onTime"] += 1
        elif r.get("checkInStatus") == "late":
            emp_map[emp_id]["late"] += 1
            emp_map[emp_id]["totalLateMin"] += r.get("lateMinutes", 0)

    # Sort by name
    sorted_emps = sorted(emp_map.values(), key=lambda x: x["name"])
    r_idx = 5
    for idx, emp in enumerate(sorted_emps, 1):
        row_vals = [
            idx,
            emp["employeeId"],
            emp["name"],
            emp["department"],
            emp["totalAttended"],
            emp["onTime"],
            emp["late"],
            emp["totalLateMin"]
        ]
        row_fill = zebra_fill if idx % 2 == 1 else None

        for c_idx, val in enumerate(row_vals, 1):
            cell = ws2.cell(row=r_idx, column=c_idx, value=val)
            cell.font = data_font
            cell.border = cell_border
            if row_fill:
                cell.fill = row_fill

            if c_idx in [1, 2, 5, 6, 7, 8]:
                cell.alignment = center_align
            else:
                cell.alignment = left_align

        ws2.row_dimensions[r_idx].height = 20
        r_idx += 1

    # Auto-fit Sheet 2
    for col in ws2.columns:
        max_len = 0
        col_letter = get_column_letter(col[0].column)
        for cell in col:
            if cell.row in [1, 2]:
                continue
            if cell.value:
                max_len = max(max_len, len(str(cell.value)))
        ws2.column_dimensions[col_letter].width = max(max_len + 4, 12)

    # 3. Save to in-memory bytes stream
    excel_buffer = io.BytesIO()
    wb.save(excel_buffer)
    excel_buffer.seek(0)

    filename_date = _get_current_wib().strftime("%Y%m%d_%H%M")
    filename = f"Laporan_Presensi_PAUD_LAB_UM_{filename_date}.xlsx"

    return Response(
        content=excel_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"'
        }
    )

@router.get("/attendance/analytics")
async def get_attendance_analytics(
    month: Optional[str] = Query(None, description="Format: YYYY-MM"),
    admin_user: TokenData = Depends(require_admin)
):
    """
    Returns high-level analytics for charts and reporting dashboards:
    - Compliance percentage
    - Department breakdown
    - Daily attendance trend over the month
    """
    db = get_firestore_client()
    target_month = month or _get_current_wib().strftime("%Y-%m")
    if db is None:
        return {"month": target_month, "totalRecords": 0, "onTimeRate": 100.0, "departments": [], "dailyTrend": []}
    docs = db.collection("attendance").stream()

    total_attended = 0
    on_time_count = 0
    late_count = 0
    dept_stats = {}
    daily_trend = {}

    for doc in docs:
        d = doc.to_dict()
        w_date = d.get("workDate", "")
        if not w_date.startswith(target_month):
            continue

        total_attended += 1
        is_on_time = d.get("checkInStatus") == "on_time"
        if is_on_time:
            on_time_count += 1
        else:
            late_count += 1

        dept = d.get("department", "Lainnya")
        if dept not in dept_stats:
            dept_stats[dept] = {"name": dept, "attended": 0, "onTime": 0, "late": 0}
        dept_stats[dept]["attended"] += 1
        if is_on_time:
            dept_stats[dept]["onTime"] += 1
        else:
            dept_stats[dept]["late"] += 1

        # Daily trend
        day_key = w_date[8:]  # DD
        if day_key not in daily_trend:
            daily_trend[day_key] = {"day": day_key, "date": w_date, "onTime": 0, "late": 0}
        if is_on_time:
            daily_trend[day_key]["onTime"] += 1
        else:
            daily_trend[day_key]["late"] += 1

    rate = round((on_time_count / total_attended * 100.0), 1) if total_attended > 0 else 100.0

    return {
        "month": target_month,
        "totalRecords": total_attended,
        "onTimeCount": on_time_count,
        "lateCount": late_count,
        "onTimeRate": rate,
        "departments": list(dept_stats.values()),
        "dailyTrend": sorted(daily_trend.values(), key=lambda x: x["day"])
    }
