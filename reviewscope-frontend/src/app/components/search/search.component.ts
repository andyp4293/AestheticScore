import { Component, EventEmitter, Output } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-search',
  standalone: false,
  templateUrl: './search.component.html',
  styleUrls: ['./search.component.css']
})
export class SearchComponent {
  @Output() searchResults = new EventEmitter<any>();
  searchQuery: string = '';

  searchBusiness() {
    if (this.searchQuery.trim()) {
      console.log('Searching for:', this.searchQuery);
      // Add API call here later
    }
  }
}
