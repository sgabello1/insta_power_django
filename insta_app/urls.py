from django.urls import path
from .views import fetch_story, index  # Import fetch_story from insta_app.views

urlpatterns = [
    path("", index, name="index"),  # Home page
    path("fetch-story/", fetch_story, name="fetch_story"),
]

