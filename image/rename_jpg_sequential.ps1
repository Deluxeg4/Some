# ตั้งค่าโฟลเดอร์ที่มีไฟล์รูปภาพ (ถ้าสคริปต์อยู่ในโฟลเดอร์เดียวกับไฟล์ ไม่ต้องเปลี่ยน)
$sourceFolder = "."

# กำหนดรูปแบบการนับเริ่มต้น (เช่น 001, 002, ...)
$initialCount = 1

Write-Host "กำลังเริ่มต้นการเปลี่ยนชื่อไฟล์..."
Write-Host ""

# ค้นหาไฟล์ .jpg ทั้งหมดในโฟลเดอร์
# Sort-Object { [int]$_.BaseName } จะพยายามแปลงชื่อไฟล์ (ไม่รวมนามสกุล) เป็นตัวเลขเพื่อจัดเรียง
# หากชื่อไฟล์ไม่ใช่ตัวเลขทั้งหมด (เช่น "abc.jpg", "123.jpg") การเรียงลำดับอาจจะไม่เป็นไปตามที่คาด
# ถ้าต้องการเรียงตามชื่อเดิมแบบตัวอักษร ให้ใช้ Sort-Object Name แทน
$files = Get-ChildItem -Path $sourceFolder -Filter "*.jpg" | Sort-Object { [int]$_.BaseName }

foreach ($file in $files) {
    # สร้างชื่อไฟล์ใหม่พร้อมรูปแบบ 3 หลัก (001, 002, ...)
    # ตัวอย่าง: "00{0:D3}" - D3 หมายถึงแสดงเป็นตัวเลข 3 หลัก พร้อมเติม 0 ข้างหน้า
    $newFileName = "{0:D3}.jpg" -f $initialCount
    $newPath = Join-Path -Path $file.DirectoryName -ChildPath $newFileName

    # ตรวจสอบว่าชื่อใหม่ไม่ซ้ำกับชื่อไฟล์ปัจจุบัน
    if ($file.FullName -ne $newPath) {
        try {
            Rename-Item -Path $file.FullName -NewName $newFileName -Force
            Write-Host "เปลี่ยนชื่อ ""$($file.Name)"" เป็น ""$newFileName""" -ForegroundColor Green
        } catch {
            Write-Host "ไม่สามารถเปลี่ยนชื่อ ""$($file.Name)"" ได้: $($_.Exception.Message)" -ForegroundColor Red
        }
    } else {
        Write-Host "ข้าม ""$($file.Name)"" (ชื่อถูกต้องอยู่แล้ว หรือไม่จำเป็นต้องเปลี่ยน)." -ForegroundColor Yellow
    }

    $initialCount++
}

Write-Host ""
Write-Host "ดำเนินการเสร็จสิ้น."
Read-Host "กด Enter เพื่อปิด"