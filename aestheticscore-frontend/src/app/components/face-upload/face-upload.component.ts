import { Component } from '@angular/core'; // necessary for creating components
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { environment } from '../../../environments/environment'; 


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

  handleFile(file: File) {
    if (file.type.startsWith('image/')) { // checks if the file is an image
      this.selectedFile = file; // sets the selectedFile property to the file 
      const reader = new FileReader(); 
      reader.onload = (e) => { // this runs when reader.readAsDataURL is loaded
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const ctx = canvas.getContext('2d');
          if (ctx) {
            const size = Math.min(img.width, img.height); // get the smallest dimension
            canvas.width = size;
            canvas.height = size;
            const offsetX = (img.width - size) / 2;
            const offsetY = (img.height - size) / 2;
            ctx.drawImage(img, offsetX, offsetY, size, size, 0, 0, size, size);
            this.selectedImage = canvas.toDataURL('image/jpeg'); // sets selected image as a base64 string for the user to preview the image
          }
        };
        img.src = e.target?.result as string;
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

      this.http.post<{beauty_score: number}>(`${environment.backendUrl}/predict`, formData)
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