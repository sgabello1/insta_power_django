from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from .insta_functions_dj import story_from_article  # Import your function

@csrf_exempt
def run_script(request):
    if request.method == "POST":
        url = request.POST.get("url", "")
        num_of_words = int(request.POST.get("num_of_words", 100))
        
        if not url:
            return JsonResponse({"error": "No URL provided"}, status=400)
        
        # Call your function (modify based on your script logic)
        result = story_from_article(url, num_of_words)

        return JsonResponse({"message": "Processing started", "result": result})
    
    return JsonResponse({"error": "Invalid request"}, status=400)
