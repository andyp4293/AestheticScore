import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';

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
    if (this.selectedImage) {
      // TODO: Add your face analysis logic here
      console.log('Analyzing image...');
    }
  }
} 