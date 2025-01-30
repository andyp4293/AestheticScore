import { Component, ViewChild, ElementRef, AfterViewInit, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

declare var google: any;

interface BusinessInfo {
  name: string;
  rating: number;
  totalRatings: number;
  placeId: string;
}

@Component({
  selector: 'app-search',
  standalone: false,
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent implements AfterViewInit, OnInit {
  @ViewChild('searchInput') searchInput!: ElementRef;
  searchQuery: string = '';
  selectedBusiness: BusinessInfo | null = null;

  ngAfterViewInit() {
    this.initAutocomplete();
  }

  ngOnInit() {
    // Initialize Google Places Autocomplete
    this.initAutocomplete();
  }

  private initAutocomplete() {
    if (this.searchInput) {
      const autocomplete = new google.maps.places.Autocomplete(
        this.searchInput.nativeElement,
        { 
          types: ['establishment'],
          fields: ['name', 'place_id', 'rating', 'user_ratings_total'] // Explicitly request these fields
        }
      );

      autocomplete.addListener('place_changed', () => {
        const place = autocomplete.getPlace();
        console.log('Place details:', place); // Debug log
        
        if (place && place.place_id) {
          this.selectedBusiness = {
            name: place.name || '',
            rating: Number(place.rating) || 0,
            totalRatings: Number(place.user_ratings_total) || 0,
            placeId: place.place_id
          };
          
          console.log('Selected business:', this.selectedBusiness); // Debug log
          
          this.searchQuery = place.name || '';
          this.analyzeBusiness(place.place_id);
        }
      });
    }
  }

  analyzeBusiness(placeId?: string) {
    if (this.searchQuery.trim()) {
      console.log('Analyzing business:', this.searchQuery, 'Place ID:', placeId);
      
    }
  }
}
