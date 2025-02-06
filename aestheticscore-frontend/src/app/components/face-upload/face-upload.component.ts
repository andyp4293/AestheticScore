import { Component } from '@angular/core'; // necessary for creating components
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

// @Component is a decorator that tells a class to behave as a component
@Component({
  selector: 'app-face-upload', // this makes the html tag for the component, <app-face-upload><app-face-upload/> 
  templateUrl: './face-upload.component.html', // this is the html file for this component
  styleUrls: ['./face-upload.component.css'], // this is the css file for this component 
  standalone: true, 
  imports: [CommonModule] 
})
export class FaceUploadComponent { // creates a class for our component
  isDragging = false; // these are basically states for our component, class properties that track the state of our component
  selectedImage: string | null = null;
  selectedFile: File | null = null;
  beautyScore: number | null = null;
  isLoading = false;

  constructor(private http: HttpClient) {} // allows us to use http requests 

  onDragOver(event: DragEvent) { // event handler for when user drags an image into the component box
    event.preventDefault();
    this.isDragging = true;
  }

  onDrop(event: DragEvent) { // called when a file is dropped into the file area
    event.preventDefault();
    this.isDragging = false; 
    
    const files = event.dataTransfer?.files; // gets the dropped files from the event
    if (files?.length) {
      this.handleFile(files[0]); // calls handleFile function with the first file dropped
    }
  }

  onFileSelected(event: Event) { // called when a user browses a photo and selects it 
    const files = (event.target as HTMLInputElement).files;
    if (files?.length) {
      this.handleFile(files[0]); // calls handleFile function with the first file dropped
    }
  }

  handleFile(file: File) { // event handler
    if (file.type.startsWith('image/')) { // checks if the file is an image
      this.selectedFile = file; // sets the selectedFile property to the file 
      const reader = new FileReader(); 
      reader.onload = (e) => { // this funs when reader.readAsDataURL is loaded
        this.selectedImage = e.target?.result as string; // sets selected image as a base64 string for the user to preview the image
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please upload an image file');
    }
  }

  // runs when user clicks "Analyze" button 
  analyzeImage() { // function to send selected image to the backend server so the model can analyze it
    if (this.selectedFile)  { // function only runs if a file is selected
      this.isLoading = true; 
      const formData = new FormData(); // form data object to send files through http 
      formData.append('file', this.selectedFile);

      this.http.post<{beauty_score: number}>('http://localhost:8000/predict', formData)
        .subscribe({
          next: (response) => {
            this.beautyScore = response.beauty_score;
            this.isLoading = false;
          },
          error: (error) => {
            console.error('Error analyzing image:', error);
            this.isLoading = false;
            alert('Error analyzing image. Please try again.');
          }
        });
    }
  }
} 