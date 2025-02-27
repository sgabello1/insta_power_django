from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponse
import json
from .insta_functions_dj import story_from_article  # Import your function
from django.shortcuts import render


def index(request):
    return render(request, "insta_app/index.html")

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


@csrf_exempt  # Disable CSRF protection for testing (use proper security in production)
def fetch_story(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)  # Parse JSON request
            url = data.get("url")  # Get the URL from the request
            num_of_words = int(data.get("num_of_words", 100))  # Default to 100 words
            
            if not url:
                return JsonResponse({"error": "URL is required"}, status=400)
            
            # Call your function
            title, summary, full_text = story_from_article(url, num_of_words)
            
            return JsonResponse({
                "title": title,
                "summary": summary,
                "full_text": full_text
            })
        
        except Exception as e:
            return JsonResponse({"error": str(e)}, status=500)
    
    return JsonResponse({"error": "Invalid request method"}, status=405)

def home(request):
    return HttpResponse("<h1>Welcome to the Insta App API</h1><p>Use <code>/insta/run/</code> to process videos.</p>")
