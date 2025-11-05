## 🐛 BUG FIXED - LLM Underplanning Issue

### ❌ Vấn đề đã phát hiện:
- User: "1 câu tự luận và 25 câu trắc nghiệm" (26 câu)
- Output: Chỉ 18 câu (thiếu 8 câu!)

### ✅ Root Cause:
LLM trong `plan_node` không tuân thủ yêu cầu số lượng câu chính xác:
- Regex parse đúng: `{'essay': 1, 'multiple_choice': 25}` ✅
- LLM plan chỉ tạo: ~18 câu ❌
- Round-robin chỉ chia được 18 câu (không thể tạo thêm từ hư không)

### 🔧 Solution Implemented:

**Added Force Fix Logic** (line ~247 in quiz_generation.py):

```python
elif expected_total > 0 and total_planned_questions < expected_total:
    missing = expected_total - total_planned_questions
    logger.warning(f"🔧 APPLYING FORCE FIX: Adding {missing} missing questions...")
    
    # Add missing questions to first task
    section_tasks.tasks[0].number_of_questions += missing
    
    # Recalculate and verify
    total_planned_questions = sum(task.number_of_questions for task in section_tasks.tasks)
    if total_planned_questions == expected_total:
        logger.info(f"✅✅✅ PERFECT! Now total = expected = {expected_total}")
```

### 📋 How It Works:

**Before Fix:**
```
User: "1 tự luận, 25 trắc nghiệm" (26 câu)
LLM plans: Task1=6, Task2=6, Task3=6 (18 câu) ❌
Round-robin allocates: 18 câu ❌
Output: 18 câu (thiếu 8!)
```

**After Fix:**
```
User: "1 tự luận, 25 trắc nghiệm" (26 câu)
LLM plans: Task1=6, Task2=6, Task3=6 (18 câu)
FORCE FIX: Task1 += 8 → Task1=14, Task2=6, Task3=6 (26 câu) ✅
Round-robin allocates: 26 câu ✅
Output: 1 tự luận + 25 trắc nghiệm ✅
```

### 🧪 Testing Steps:

1. **Restart app:**
   ```powershell
   # Press Ctrl+C in running terminal
   python ui/app.py
   ```

2. **Test với input:**
   - "1 câu tự luận và 25 câu trắc nghiệm"
   - "1 câu tự luận và 21 câu trắc nghiệm"

3. **Verify logs xuất hiện:**
   ```
   📊 Total planned questions FROM LLM: 18
   ❌ LLM UNDERPLANNED: Planned 18 but expected 26!
   🔧 APPLYING FORCE FIX: Adding missing questions...
   ✅ Added 8 questions to Task 1
   📊 Total after force fix: 26
   ✅✅✅ PERFECT! Now total = expected = 26
   === TYPE DISTRIBUTION ALGORITHM ===
   Question pool to distribute: ['essay', 'MC', 'MC', ...(26 total)]
   Total allocated: 26, Expected: 26
   ✅ Perfect match! Allocated == Required
   ```

4. **Count output:** Phải nhận được ĐÚNG 1 tự luận + 25 trắc nghiệm!

### 🎯 Expected Results:
- ✅ Regex parse: Đúng
- ✅ Force fix: Điều chỉnh total cho khớp
- ✅ Round-robin: Phân bổ đúng 26 câu
- ✅ Output: 1 essay + 25 MC

### 📊 Status:
- [x] Debug logging added
- [x] Force fix implemented
- [ ] Testing in progress (restart app required)
- [ ] Verify output count matches request

---
**Next Action:** RESTART APP và test ngay! 🚀
