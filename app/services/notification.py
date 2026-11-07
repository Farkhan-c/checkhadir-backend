"""
Automatic Attendance Notification Service (Email SMTP & WhatsApp Gateway)
Dispatches confirmation receipts asynchronously via FastAPI BackgroundTasks / thread pool
after Check-In and Check-Out records are persisted in Cloud Firestore.
All network I/O is wrapped in defensive try-except blocks so notification errors
never interrupt or roll back an employee's verified attendance transaction.
"""

import asyncio
from email.message import EmailMessage
from email.utils import formataddr
import html
import logging
import smtplib
from typing import Optional, Dict

import httpx

from app.core.config import settings
from app.models.auth import normalize_indonesian_phone

logger = logging.getLogger(__name__)


def format_whatsapp_attendance_message(
    nama: str,
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
) -> str:
    """
    Formats the standardized WhatsApp text notification message.
    Exact template:
    "Halo {nama}, presensi {type} Anda pada {date} pukul {time} WIB berhasil dicatat. Status: {status}."
    """
    return (
        f"Halo {nama}, presensi {attendance_type} Anda pada {work_date} "
        f"pukul {time_wib} WIB berhasil dicatat. Status: {status_label}."
    )


def _normalize_target_phone(phone: str) -> str:
    """Normalizes phone number to 62... format when valid; falls back to stripped digits."""
    try:
        return normalize_indonesian_phone(phone)
    except ValueError:
        return "".join(ch for ch in phone if ch.isdigit())


def send_attendance_email(
    recipient_email: Optional[str],
    employee_name: str,
    niy: str,
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
    notes: Optional[str] = None,
) -> bool:
    """
    Sends an HTML + Plaintext attendance confirmation email via Python standard `smtplib`.
    Reads SMTP_HOST, SMTP_PORT, SMTP_USER, and SMTP_PASSWORD from `settings`.
    Returns True if sent, False if skipped or failed (never raises).
    """
    if not recipient_email:
        logger.info("Skipping email notification: employee has no email address.")
        return False

    smtp_user = (settings.SMTP_USER or "").strip()
    smtp_password = settings.SMTP_PASSWORD or ""
    smtp_host = (settings.SMTP_HOST or "smtp.gmail.com").strip()
    smtp_port = int(settings.SMTP_PORT or 587)

    if not smtp_user or not smtp_password:
        logger.info(
            "Skipping email notification: SMTP_USER or SMTP_PASSWORD is not configured."
        )
        return False

    notes_display = (notes or "").strip() or "-"
    timestamp_wib = f"{work_date} pukul {time_wib} WIB"

    plain_body = (
        f"Halo {employee_name},\n\n"
        f"Presensi kehadiran Anda di {settings.ORGANIZATION_NAME} berhasil dicatat.\n\n"
        f"Rincian Presensi:\n"
        f"- Nama Karyawan     : {employee_name}\n"
        f"- NIY / NIP         : {niy}\n"
        f"- Jenis Presensi    : {attendance_type}\n"
        f"- Waktu Presensi    : {timestamp_wib}\n"
        f"- Status Kehadiran  : {status_label}\n"
        f"- Catatan Kehadiran : {notes_display}\n\n"
        f"Pesan ini dikirim secara otomatis oleh sistem {settings.APP_NAME}.\n"
    )

    safe_name = html.escape(employee_name)
    safe_niy = html.escape(niy)
    safe_type = html.escape(attendance_type)
    safe_time = html.escape(timestamp_wib)
    safe_status = html.escape(status_label)
    safe_notes = html.escape(notes_display)
    safe_org = html.escape(settings.ORGANIZATION_NAME)

    badge_bg = "#ecfdf5" if "Tepat Waktu" in status_label else "#fffbeb"
    badge_fg = "#047857" if "Tepat Waktu" in status_label else "#b45309"

    html_body = f"""<!DOCTYPE html>
<html lang="id">
<head><meta charset="utf-8"></head>
<body style="font-family: Arial, Helvetica, sans-serif; background-color: #f8fafc; margin: 0; padding: 24px; color: #0f172a;">
  <div style="max-width: 560px; margin: 0 auto; background: #ffffff; border: 1px solid #e2e8f0; border-radius: 14px; overflow: hidden;">
    <div style="background: #0f172a; color: #ffffff; padding: 20px 24px;">
      <div style="font-size: 11px; color: #93c5fd; font-weight: bold; letter-spacing: 0.08em; text-transform: uppercase;">
        {safe_org}
      </div>
      <h2 style="margin: 6px 0 0; font-size: 18px;">Bukti Presensi {safe_type} — CheckHadir.Id</h2>
    </div>
    <div style="padding: 24px; font-size: 14px; line-height: 1.6;">
      <p style="margin-top: 0;">Halo <strong>{safe_name}</strong>,</p>
      <p>Presensi Anda telah terverifikasi secara biometrik dan berhasil tersimpan di sistem.</p>
      <table style="width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px;">
        <tr>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b; width: 42%;">Nama Karyawan</td>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; font-weight: bold;">{safe_name}</td>
        </tr>
        <tr>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b;">NIY / NIP</td>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; font-family: monospace; font-weight: bold;">{safe_niy}</td>
        </tr>
        <tr>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b;">Jenis Presensi</td>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; font-weight: bold;">{safe_type}</td>
        </tr>
        <tr>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b;">Waktu Presensi (WIB)</td>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; font-weight: bold;">{safe_time}</td>
        </tr>
        <tr>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9; color: #64748b;">Status</td>
          <td style="padding: 8px 10px; border-bottom: 1px solid #f1f5f9;">
            <span style="display: inline-block; padding: 3px 10px; border-radius: 999px; background: {badge_bg}; color: {badge_fg}; font-weight: bold; font-size: 12px;">
              {safe_status}
            </span>
          </td>
        </tr>
        <tr>
          <td style="padding: 8px 10px; color: #64748b;">Catatan Kehadiran</td>
          <td style="padding: 8px 10px;">{safe_notes}</td>
        </tr>
      </table>
      <p style="font-size: 12px; color: #64748b; margin-bottom: 0;">
        Email ini dibuat otomatis oleh layanan notifikasi {html.escape(settings.APP_NAME)}. Mohon tidak membalas email ini.
      </p>
    </div>
  </div>
</body>
</html>"""

    msg = EmailMessage()
    msg["Subject"] = f"[CheckHadir.Id] Bukti Presensi {attendance_type} - {employee_name} ({work_date})"
    msg["From"] = formataddr((settings.SMTP_FROM_NAME, smtp_user))
    msg["To"] = recipient_email
    msg.set_content(plain_body)
    msg.add_alternative(html_body, subtype="html")

    try:
        if smtp_port == 465:
            with smtplib.SMTP_SSL(smtp_host, smtp_port, timeout=10) as server:
                server.login(smtp_user, smtp_password)
                server.send_message(msg)
        else:
            with smtplib.SMTP(smtp_host, smtp_port, timeout=10) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

        logger.info(f"Attendance email notification sent to {recipient_email} ({attendance_type}).")
        return True
    except Exception as exc:
        logger.warning(f"Failed to send attendance email to {recipient_email}: {exc}")
        return False


async def send_attendance_email_async(
    recipient_email: Optional[str],
    employee_name: str,
    niy: str,
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
    notes: Optional[str] = None,
) -> bool:
    """Async thread-pool wrapper around `send_attendance_email`."""
    return await asyncio.to_thread(
        send_attendance_email,
        recipient_email,
        employee_name,
        niy,
        attendance_type,
        work_date,
        time_wib,
        status_label,
        notes,
    )


def send_attendance_whatsapp(
    phone: Optional[str],
    employee_name: str,
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
) -> bool:
    """
    Sends a WhatsApp text notification via HTTP POST gateway (generic JSON / Fonnte compatible).
    Reads WA_GATEWAY_URL and WA_API_TOKEN from `settings`.
    Returns True if HTTP 2xx is returned, False if skipped or failed (never raises).
    """
    if not phone or not phone.strip():
        logger.info("Skipping WhatsApp notification: employee has no phone/WA number.")
        return False

    gateway_url = (settings.WA_GATEWAY_URL or "").strip()
    api_token = (settings.WA_API_TOKEN or "").strip()

    if not gateway_url or not api_token:
        logger.info(
            "Skipping WhatsApp notification: WA_GATEWAY_URL or WA_API_TOKEN is not configured."
        )
        return False

    normalized_phone = _normalize_target_phone(phone)
    if not normalized_phone:
        logger.warning("Skipping WhatsApp notification: phone number could not be normalized.")
        return False

    message_text = format_whatsapp_attendance_message(
        nama=employee_name,
        attendance_type=attendance_type,
        work_date=work_date,
        time_wib=time_wib,
        status_label=status_label,
    )

    headers = {
        "Authorization": api_token,
        "Content-Type": "application/json",
    }
    payload = {
        "target": normalized_phone,
        "phone": normalized_phone,
        "message": message_text,
        "countryCode": "62",
    }

    try:
        with httpx.Client(timeout=10.0) as http_client:
            resp = http_client.post(gateway_url, json=payload, headers=headers)
            if 200 <= resp.status_code < 300:
                logger.info(
                    f"Attendance WhatsApp notification sent to {normalized_phone} ({attendance_type})."
                )
                return True
            logger.warning(
                f"WhatsApp gateway returned HTTP {resp.status_code} for {normalized_phone}."
            )
            return False
    except Exception as exc:
        logger.warning(f"Failed to send WhatsApp notification to {normalized_phone}: {exc}")
        return False


async def send_attendance_whatsapp_async(
    phone: Optional[str],
    employee_name: str,
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
) -> bool:
    """Async thread-pool wrapper around `send_attendance_whatsapp`."""
    return await asyncio.to_thread(
        send_attendance_whatsapp,
        phone,
        employee_name,
        attendance_type,
        work_date,
        time_wib,
        status_label,
    )


def dispatch_attendance_notifications(
    *,
    employee_name: str,
    niy: str,
    email: Optional[str],
    phone: Optional[str],
    attendance_type: str,
    work_date: str,
    time_wib: str,
    status_label: str,
    notes: Optional[str] = None,
) -> Dict[str, bool]:
    """
    Background task entrypoint invoked by FastAPI `BackgroundTasks` (executed in Starlette's thread pool).
    Isolates Email and WhatsApp delivery independently so an error in either channel never propagates.
    """
    email_sent = False
    wa_sent = False

    try:
        email_sent = send_attendance_email(
            recipient_email=email,
            employee_name=employee_name,
            niy=niy,
            attendance_type=attendance_type,
            work_date=work_date,
            time_wib=time_wib,
            status_label=status_label,
            notes=notes,
        )
    except Exception as exc:
        logger.warning(f"Unhandled error in email notification task: {exc}")

    try:
        wa_sent = send_attendance_whatsapp(
            phone=phone,
            employee_name=employee_name,
            attendance_type=attendance_type,
            work_date=work_date,
            time_wib=time_wib,
            status_label=status_label,
        )
    except Exception as exc:
        logger.warning(f"Unhandled error in WhatsApp notification task: {exc}")

    return {"emailSent": email_sent, "whatsappSent": wa_sent}
