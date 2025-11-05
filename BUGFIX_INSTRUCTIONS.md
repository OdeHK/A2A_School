# 🐛 BUG FIX: Thiếu 2 câu hỏi (20 thay vì 22)

## 📋 Vấn đề
User yêu cầu: **"1 câu tự luận và 21 câu trắc nghiệm"** (22 câu)  
Output nhận được: **20 câu trắc nghiệm** (thiếu 1 tự luận + 1 trắc nghiệm)

## 🔍 Nguyên nhân
**LLM trong plan_node KHÔNG TUÂN THỦ số lượng câu hỏi chính xác**

- Regex parse đúng: `{'essay': 1, 'multiple_choice': 21}` ✅
- Round-robin algorithm đúng ✅  
- **NHƯNG**: LLM plan chỉ tạo 20 câu thay vì 22 câu ❌
- Round-robin chỉ có thể phân bổ số câu mà LLM đã plan (garbage in, garbage out)

## ✅ Giải pháp

### Bước 1: Thêm Debug Logging (ĐÃ XONG)
Tôi đã thêm logging để track:
```
✅ Found essay requirement: +1 (total: 1)
✅ Found multiple choice requirement: +21 (total: 21)
📊 Expected total from types: 22 questions
📊 Total planned questions FROM LLM: ??? ← Kiểm tra số này!
❌ LLM UNDERPLANNED: Planned 20 but expected 22! ← Nếu xuất hiện = bug!
```

### Bước 2: Cải thiện LLM Prompt (CẦN LÀM)
Thêm vào `plan_prompt` (line ~112):

```python
("human", 
 "⚠️ CRITICAL REQUIREMENT - EXACT QUESTION COUNT:\n"
 "- Parse the teacher's request for EXACT question counts\n"
 "- If teacher says '1 câu tự luận và 21 câu trắc nghiệm', you MUST plan for EXACTLY 22 questions total (1 + 21)\n"
 "- The sum of number_of_questions across ALL tasks MUST EQUAL the total requested\n"
 "- DO NOT under-plan or over-plan. Be PRECISE!\n\n"
 # ... rest of prompt
)
```

### Bước 3: Fallback Fix - Force Correct Total
Nếu LLM vẫn sai, thêm logic điều chỉnh SAU KHI parse type requirements:

```python
# After line ~213 (nơi có type_requirements)
if type_requirements and total_planned_questions != expected_total:
    logger.warning(f"⚠️ LLM planned {total_planned_questions} but expected {expected_total}")
    logger.warning(f"⚠️ FORCE CORRECTING: Adjusting first task...")
    
    # Adjust first task to compensate
    if section_tasks.tasks:
        diff = expected_total - total_planned_questions
        section_tasks.tasks[0].number_of_questions += diff
        logger.info(f"✅ Adjusted Task 1: +{diff} questions (new total: {section_tasks.tasks[0].number_of_questions})")
        
        # Recalculate
        total_planned_questions = sum(task.number_of_questions for task in section_tasks.tasks)
        logger.info(f"📊 New total after adjustment: {total_planned_questions}")
```

## 🧪 Cách Test

### Test 1: Chạy lại với logging mới
```bash
# Restart app
python ui/app.py

# Nhập: "1 câu tự luận và 21 câu trắc nghiệm"
# Tìm trong log:
# - "📊 Total planned questions FROM LLM: ??"
# - "❌ LLM UNDERPLANNED:" (nếu có = confirmed bug)
```

### Test 2: Verify với script
```bash
python test_regex_debug.py
# Should show: {'essay': 1, 'multiple_choice': 21} Total: 22 ✅
```

## 📊 Expected Log Output (Sau khi fix)
```
🔍 Regex matches found: [('1', 'tự luận'), ('21', 'trắc nghiệm')]
✅ Found essay requirement: +1 (total: 1)
✅ Found multiple choice requirement: +21 (total: 21)
📊 Final type requirements: {'essay': 1, 'multiple_choice': 21}
📊 Expected total from types: 22 questions
📊 Total planned questions FROM LLM: 22  ← MUST BE 22!
=== TYPE DISTRIBUTION ALGORITHM ===
Type requirements: {'essay': 1, 'multiple_choice': 21}
Question pool to distribute: ['essay', 'MC', 'MC', ..., 'MC']  ← 22 items
Total allocated: 22, Expected: 22
✅ Allocated by type: {'essay': 1, 'multiple_choice': 21}
📌 Required by type: {'essay': 1, 'multiple_choice': 21}
✅ Perfect match! Allocated == Required
```

## 🎯 Hành động tiếp theo
1. ✅ **Chạy app lại và test** với input "1 câu tự luận và 21 câu trắc nghiệm"
2. ✅ **Gửi log cho tôi**, đặc biệt dòng "Total planned questions FROM LLM: ??"
3. ⏳ Nếu vẫn sai, tôi sẽ implement **Fallback Fix** (Bước 3 ở trên)

## 📝 Ghi chú
- Tôi ĐÃ thêm debug logging vào file
- Tôi CHƯA thêm fallback fix (chờ xác nhận bug từ log)
- Regex hoạt động hoàn hảo ✅
- Round-robin hoạt động hoàn hảo ✅  
- **Vấn đề nằm ở LLM plan_node** ❌
