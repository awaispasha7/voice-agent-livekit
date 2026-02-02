/**
 * Agent Profile Service
 * Manages human agent identity (name, ID) for the dashboard
 */

import { Injectable } from '@angular/core';
import { BehaviorSubject, Observable } from 'rxjs';

export interface AgentProfile {
  agentId: string;
  agentName: string;
  timestamp: number;
}

@Injectable({
  providedIn: 'root'
})
export class AgentProfileService {
  private readonly STORAGE_KEY = 'alive5_agent_profile';
  private profileSubject = new BehaviorSubject<AgentProfile | null>(null);
  public profile$: Observable<AgentProfile | null> = this.profileSubject.asObservable();

  constructor() {
    this.loadProfile();
  }

  /**
   * Load agent profile from localStorage
   */
  private loadProfile(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY);
      if (stored) {
        const profile = JSON.parse(stored) as AgentProfile;
        this.profileSubject.next(profile);
        console.log('[AgentProfile] Loaded profile:', profile.agentName);
      }
    } catch (error) {
      console.error('[AgentProfile] Failed to load profile:', error);
    }
  }

  /**
   * Set agent profile
   */
  setProfile(name: string): AgentProfile {
    const profile: AgentProfile = {
      agentId: `agent_${Math.random().toString(36).substring(2, 9)}`,
      agentName: name.trim(),
      timestamp: Date.now()
    };

    try {
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(profile));
      this.profileSubject.next(profile);
      console.log('[AgentProfile] Profile saved:', profile);
    } catch (error) {
      console.error('[AgentProfile] Failed to save profile:', error);
    }

    return profile;
  }

  /**
   * Get current profile
   */
  getProfile(): AgentProfile | null {
    return this.profileSubject.value;
  }

  /**
   * Check if profile exists
   */
  hasProfile(): boolean {
    return this.profileSubject.value !== null;
  }

  /**
   * Clear profile (logout)
   */
  clearProfile(): void {
    try {
      localStorage.removeItem(this.STORAGE_KEY);
      this.profileSubject.next(null);
      console.log('[AgentProfile] Profile cleared');
    } catch (error) {
      console.error('[AgentProfile] Failed to clear profile:', error);
    }
  }

  /**
   * Update agent name
   */
  updateName(newName: string): void {
    const current = this.getProfile();
    if (current) {
      this.setProfile(newName);
    }
  }
}
