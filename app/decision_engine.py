# -*- coding: utf-8 -*-

def normalize_label(x):
    return str(x).strip().lower() if x is not None else ""

def choose_segment(total_messages=0, negative_count=0, complaint_count=0, inactive_days=0):
    total_messages = int(total_messages or 0)
    negative_count = int(negative_count or 0)
    complaint_count = int(complaint_count or 0)
    inactive_days = int(inactive_days or 0)

    if inactive_days >= 30:
        return "Lost Customer"
    if negative_count >= 3 or complaint_count >= 3:
        return "At-Risk"
    if total_messages >= 20 and negative_count <= 2:
        return "VIP"
    if total_messages >= 5:
        return "Active"
    return "Regular"

def choose_action(sentiment, category, segment):
    s = normalize_label(sentiment)
    c = str(category or "")
    seg = str(segment or "")

    # ---------- 1) เคสร้ายแรง/ร้องเรียนหนัก (ตรวจก่อนเสมอ) ----------
    if s == "negative" and ("มีแมลง" in c or "เจอสิ่งแปลกปลอม" in c or "อาหารเป็นพิษ" in c or "บูด" in c or "เสีย" in c):
        return "apology_quality_escalate"          # คุณภาพอาหาร - ร้องเรียนหนัก, escalate ทุกเซกเมนต์

    if s == "negative" and ("พนักงานหยาบคาย" in c or "บริการแย่" in c or "ไม่สุภาพ" in c):
        return "apology_staff_escalate"            # ร้องเรียนพนักงาน

    if s == "negative" and ("สกปรก" in c or "กลิ่นเหม็น" in c or "แมลงสาบ" in c):
        return "apology_cleanliness_escalate"      # ความสะอาด

    if s == "negative" and ("แอร์" in c or "ร้อนมาก" in c or "อุณหภูมิ" in c):
        return "apology_temperature_escalate"

    if s == "negative" and ("ห้องน้ำ" in c):
        return "apology_restroom_escalate"

    if s == "negative" and ("บิล" in c and ("ผิด" in c or "ยอดไม่ตรง" in c or "เก็บเงินเกิน" in c)):
        return "apology_billing_escalate"

    if s == "negative" and ("แอพ" in c or "ระบบ" in c or "เว็บค้าง" in c):
        return "apology_system_issue"

    if s == "negative" and ("บัตรสมาชิก" in c or "แต้ม" in c):
        return "apology_membership_issue"

    # ---------- 2) จัดส่ง / ออเดอร์ผิดพลาด ----------
    if s == "negative" and ("จัดส่ง" in c or "Delivery" in c or "ช้า" in c or "อาหารไม่ตรง" in c):
        if seg in ["VIP", "At-Risk"]:
            return "apology_coupon_escalate"
        return "apology_check_order"

    if s == "negative" and ("ไม่ครบ" in c or "ขาด" in c or "ผิดออเดอร์" in c or "สั่งผิด" in c):
        if seg in ["VIP", "At-Risk"]:
            return "apology_coupon_escalate"
        return "apology_wrong_order"

    # ---------- 3) ทั่วไป negative ----------
    if s == "negative":
        return "apology_escalate"

    # ---------- 4) สอบถามข้อมูลทั่วไป ----------
    if "โปรโมชั่น" in c or "Promotion" in c or "โปร" in c:
        return "send_promotion"

    if "เมนู" in c or "Menu" in c:
        return "send_menu"

    if "จอง" in c or "Reservation" in c:
        return "reservation_support"

    if "เวลาเปิด" in c or "เวลาทำการ" in c:
        return "send_hours"

    if "แผนที่" in c or "สถานที่" in c or "location" in c:
        return "send_location"

    if "ราคา" in c or "บิล" in c:
        return "send_price_info"

    if "ชำระเงิน" in c or "payment" in c:
        return "send_payment_info"

    if "แพ้" in c or "มังสวิรัติ" in c or "ฮาลาล" in c:
        return "allergy_info_caution"

    if "ใบกำกับภาษี" in c or "ออกบิล" in c:
        return "send_tax_invoice_info"

    if "สมาชิก" in c or "สะสมแต้ม" in c:
        return "send_membership_info"

    if "จัดเลี้ยง" in c or "อีเวนต์" in c or "catering" in c:
        return "catering_support"

    if "สมัครงาน" in c or "รับสมัคร" in c:
        return "job_application_info"

    if "สาขา" in c or "ทำเล" in c or "อยู่ตรงไหน" in c or "BTS" in c or "MRT" in c:
        return "branch_location_info"

    if "ที่จอดรถ" in c or "จอดรถ" in c or "ลานจอด" in c:
        return "parking_info"

    if "คิวรอโต๊ะ" in c:
        return "waitlist_info"

    if "WiFi" in c or "ไวไฟ" in c:
        return "wifi_info"

    if "สัตว์เลี้ยง" in c:
        return "pet_friendly_info"

    if "เด็ก" in c:
        return "kids_facility_info"

    if "ห้องส่วนตัว" in c or "VIP" in c:
        return "private_room_info"

    if "Corkage" in c:
        return "corkage_info"

    if "ปรับระดับความเผ็ด" in c:
        return "spice_customization_info"

    if "บุฟเฟ่ต์" in c:
        return "buffet_info"

    if "ข้อเสนอแนะ" in c:
        return "feedback_thank_you"

    if "เครื่องดื่มแอลกอฮอล์" in c:
        return "alcohol_menu_info"

    if "ดนตรีสด" in c or "กิจกรรม" in c:
        return "event_info"

    if "พื้นที่จัดส่ง" in c:
        return "delivery_area_info"

    if "ยอดสั่งขั้นต่ำ" in c:
        return "min_order_info"

    if "คืนเงิน" in c or "ยกเลิก" in c or "refund" in c:
        if seg in ["VIP", "At-Risk"]:
            return "refund_priority_support"
        return "refund_cancel_support"

    # ---------- 5) รสชาติ (แยกตาม sentiment เพราะอาจเป็นได้ทั้งบวก/ลบ) ----------
    if "รสชาติ" in c:
        return "thank_you" if s == "positive" else "apology_escalate"

    # ---------- 6) positive ทั่วไป ----------
    if s == "positive":
        return "thank_you"

    if "ชมบรรยากาศร้าน" in c or "ชมพนักงาน" in c:
        return "thank_you"

    return "general_support"


REPLY_TEMPLATES = {
    "apology_coupon_escalate": "ขออภัยอย่างสูงค่ะ ทางร้านจะรีบตรวจสอบรายการนี้ให้ทันที และขอมอบคูปองส่วนลดสำหรับการสั่งครั้งถัดไปค่ะ",
    "apology_check_order": "ขออภัยในความไม่สะดวกค่ะ ทางร้านจะรีบตรวจสอบรายการอาหาร/การจัดส่งให้ทันทีค่ะ",
    "apology_wrong_order": "ขออภัยค่ะที่ออเดอร์ไม่ครบถ้วน ทางร้านจะตรวจสอบและจัดส่งของที่ขาดให้ทันทีค่ะ รบกวนแจ้งเลขที่ออเดอร์ด้วยนะคะ",
    "apology_quality_escalate": "ขออภัยอย่างสูงค่ะ เรื่องนี้เป็นเรื่องสำคัญมาก ทางร้านจะส่งต่อให้ทีมงานตรวจสอบด่วนที่สุด รบกวนขอรายละเอียดเพิ่มเติม (เช่น รูปภาพ) เพื่อให้เราปรับปรุงและดูแลคุณได้ดีขึ้นค่ะ",
    "apology_staff_escalate": "ขออภัยอย่างสูงค่ะสำหรับประสบการณ์ที่ไม่ดี ทางร้านจะนำเรื่องนี้ไปแจ้งและอบรมทีมงานเพื่อไม่ให้เกิดขึ้นอีกค่ะ",
    "apology_cleanliness_escalate": "ขออภัยค่ะ ทางร้านให้ความสำคัญกับความสะอาดเป็นอย่างมาก จะรีบตรวจสอบและปรับปรุงทันทีค่ะ",
    "apology_escalate": "ขออภัยอย่างสูงค่ะ ทางร้านรับทราบปัญหาแล้ว และจะส่งเรื่องให้ผู้จัดการตรวจสอบเพื่อปรับปรุงบริการค่ะ",
    "send_promotion": "ตอนนี้ทางร้านมีโปรโมชั่นพิเศษค่ะ สนใจโปรโมชั่นหรือเมนูแนะนำเพิ่มเติมไหมคะ",
    "send_menu": "ยินดีค่ะ ทางร้านมีเมนูแนะนำหลายรายการ เดี๋ยวส่งรายละเอียดเมนูยอดนิยมให้ลูกค้าค่ะ",
    "reservation_support": "ยินดีค่ะ ลูกค้าต้องการจองโต๊ะวันและเวลาใด แจ้งจำนวนที่นั่งได้เลยค่ะ",
    "send_hours": "ทางร้านเปิดให้บริการทุกวันค่ะ รบกวนแจ้งวันที่สนใจ เดี๋ยวเช็คเวลาเปิด-ปิดที่แน่นอนให้นะคะ",
    "send_location": "ส่งพิกัด/แผนที่ร้านให้นะคะ หากต้องการเส้นทางหรือจุดจอดรถ แจ้งได้เลยค่ะ",
    "send_price_info": "รบกวนแจ้งเมนูที่สนใจ ทางร้านจะแจ้งราคาที่แน่นอนให้ค่ะ",
    "send_payment_info": "ทางร้านรับชำระได้หลายช่องทางค่ะ ทั้งเงินสด โอน พร้อมเพย์ และบัตรเครดิต",
    "allergy_info_caution": "ขอบคุณที่แจ้งค่ะ รบกวนระบุอาการแพ้หรือข้อจำกัดด้านอาหารให้ชัดเจน ทางร้านจะแนะนำเมนูที่เหมาะสมและระวังส่วนผสมให้เป็นพิเศษค่ะ",
    "send_tax_invoice_info": "รบกวนแจ้งชื่อ-ที่อยู่ และเลขผู้เสียภาษีสำหรับออกใบกำกับภาษีเต็มรูปแบบนะคะ",
    "send_membership_info": "ทางร้านมีระบบสะสมแต้มค่ะ ทุกการสั่งซื้อสามารถสะสมแต้มแลกส่วนลด/ของรางวัลได้ สนใจสมัครสมาชิกไหมคะ",
    "catering_support": "ยินดีให้บริการจัดเลี้ยงค่ะ รบกวนแจ้งวันที่ จำนวนคน และงบประมาณโดยประมาณ ทางร้านจะจัดทำใบเสนอราคาให้ค่ะ",
    "job_application_info": "ขอบคุณที่สนใจร่วมงานกับเราค่ะ รบกวนแจ้งตำแหน่งที่สนใจ ทางร้านจะส่งรายละเอียดการสมัครให้ค่ะ",
    "refund_priority_support": "ขออภัยในความไม่สะดวกค่ะ ทางร้านจะดำเนินการคืนเงิน/ยกเลิกออเดอร์ให้โดยเร็วที่สุดเป็นกรณีพิเศษค่ะ",
    "refund_cancel_support": "รับทราบค่ะ ทางร้านจะดำเนินการคืนเงิน/ยกเลิกออเดอร์ให้ รบกวนแจ้งเลขที่ออเดอร์ด้วยนะคะ",
    "thank_you": "ขอบคุณมากค่ะ ทางร้านดีใจที่ลูกค้าประทับใจ หวังว่าจะได้ดูแลลูกค้าอีกครั้งนะคะ",
    "general_support": "ขอบคุณที่ติดต่อมาค่ะ ทางร้านยินดีให้บริการ ต้องการสอบถามข้อมูลเพิ่มเติมเรื่องใดแจ้งได้เลยค่ะ",
    "branch_location_info": "ขอบคุณที่สอบถามค่ะ ทางร้านมีหลายสาขา สามารถดูที่ตั้งและแผนที่แต่ละสาขาได้ทางลิงก์ด้านล่างนี้ค่ะ",
    "parking_info": "ร้านมีที่จอดรถให้บริการค่ะ หากที่จอดเต็มสามารถสอบถามพนักงานหน้าร้านเพื่อแนะนำจุดจอดใกล้เคียงได้เลยค่ะ",
    "apology_temperature_escalate": "ขออภัยในความไม่สบายค่ะ ทางร้านจะรีบตรวจสอบและปรับอุณหภูมิห้องให้เหมาะสมโดยเร็วที่สุดค่ะ",
    "apology_restroom_escalate": "ขออภัยค่ะ ทางร้านจะให้ทีมงานตรวจสอบและทำความสะอาดห้องน้ำทันทีค่ะ",
    "apology_billing_escalate": "ขออภัยค่ะสำหรับความผิดพลาดเรื่องยอดเงิน ทางร้านจะตรวจสอบบิลและแก้ไขให้ถูกต้องโดยเร็วที่สุดค่ะ รบกวนแจ้งเลขที่ออเดอร์ด้วยนะคะ",
    "apology_system_issue": "ขออภัยในความไม่สะดวกค่ะ ทางทีมงานกำลังเร่งแก้ไขปัญหาระบบ รบกวนลองใหม่อีกครั้งหรือแจ้งเจ้าหน้าที่โดยตรงได้เลยค่ะ",
    "apology_membership_issue": "ขออภัยค่ะ ทางร้านจะตรวจสอบข้อมูลบัตรสมาชิก/แต้มสะสมให้ รบกวนแจ้งเบอร์โทรหรือรหัสสมาชิกด้วยนะคะ",
    "waitlist_info": "ตอนนี้คิวประมาณ [ระบุเวลา] นาทีค่ะ ลูกค้าสามารถแจ้งชื่อ-เบอร์โทรไว้ล่วงหน้าเพื่อจองคิวได้เลยค่ะ",
    "wifi_info": "รหัส WiFi ของร้านคือ [ระบุรหัส] ค่ะ สามารถสอบถามพนักงานหน้าร้านเพิ่มเติมได้เลยค่ะ",
    "pet_friendly_info": "ร้านอนุญาตให้พาสัตว์เลี้ยงเข้าได้ค่ะ (ตามพื้นที่ที่กำหนด) รบกวนพาน้องๆ อยู่ในสายจูงและดูแลตามมารยาทด้วยนะคะ",
    "kids_facility_info": "ร้านมีเก้าอี้เด็กและเมนูสำหรับเด็กให้บริการค่ะ สามารถแจ้งพนักงานเมื่อมาถึงร้านได้เลยค่ะ",
    "private_room_info": "ร้านมีห้องส่วนตัว/ห้อง VIP ให้บริการค่ะ รบกวนแจ้งจำนวนคนและวันเวลาที่สนใจใช้บริการนะคะ",
    "corkage_info": "ลูกค้าสามารถนำเครื่องดื่ม/เค้กมาเองได้ค่ะ ทางร้านมีค่า Corkage [ระบุราคา] ต่อขวด/ชิ้นค่ะ",
    "spice_customization_info": "ทางร้านสามารถปรับระดับความเผ็ดหรือรสชาติตามที่ลูกค้าต้องการได้ค่ะ รบกวนแจ้งรายละเอียดตอนสั่งได้เลยค่ะ",
    "buffet_info": "ร้านมีโปรบุฟเฟ่ต์ให้บริการค่ะ รบกวนแจ้งวันที่สนใจ ทางร้านจะแจ้งราคาและเงื่อนไขให้ค่ะ",
    "feedback_thank_you": "ขอบคุณมากค่ะสำหรับข้อเสนอแนะ ทางร้านจะนำไปปรับปรุงบริการให้ดียิ่งขึ้นค่ะ",
    "alcohol_menu_info": "ร้านมีเครื่องดื่มแอลกอฮอล์ให้บริการค่ะ สนใจดูเมนูเครื่องดื่มเพิ่มเติมไหมคะ",
    "event_info": "ร้านมีกิจกรรม/ดนตรีสดเป็นบางช่วงค่ะ รบกวนแจ้งวันที่สนใจ ทางร้านจะเช็คตารางให้นะคะ",
    "delivery_area_info": "รบกวนแจ้งที่อยู่หรือพื้นที่ที่ต้องการจัดส่ง ทางร้านจะเช็คว่าอยู่ในพื้นที่บริการหรือไม่ค่ะ",
    "min_order_info": "ยอดสั่งขั้นต่ำสำหรับบริการเดลิเวอรี่คือ [ระบุยอด] บาทค่ะ",
}

def make_reply(action):
    return REPLY_TEMPLATES.get(action, REPLY_TEMPLATES["general_support"])
