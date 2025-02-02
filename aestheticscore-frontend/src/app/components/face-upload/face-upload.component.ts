import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';

@Component({
  selector: 'app-face-upload',
  templateUrl: './face-upload.component.html',
  styleUrls: ['./face-upload.component.css'],
  standalone: true,
  imports: [CommonModule]
})
export class FaceUploadComponent {
  isDragging = false;
  selectedImage: string | null = null;
  selectedFile: File | null = null;
  beautyScore: number | null = null;
  isLoading = false;

  constructor(private http: HttpClient) {}

  onDragOver(event: DragEvent) {
    event.preventDefault();
    this.isDragging = true;
  }

  onDrop(event: DragEvent) {
    event.preventDefault();
    this.isDragging = false;
    
    const files = event.dataTransfer?.files;
    if (files?.length) {
      this.handleFile(files[0]);
    }
  }

  onFileSelected(event: Event) {
    const files = (event.target as HTMLInputElement).files;
    if (files?.length) {
      this.handleFile(files[0]);
    }
  }

  handleFile(file: File) {
    if (file.type.startsWith('image/')) {
      this.selectedFile = file;
      const reader = new FileReader();
      reader.onload = (e) => {
        this.selectedImage = e.target?.result as string;
      };
      reader.readAsDataURL(file);
    } else {
      alert('Please upload an image file');
    }
  }

  analyzeImage() {
    if (this.selectedFile) {
      this.isLoading = true;
      const formData = new FormData();
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