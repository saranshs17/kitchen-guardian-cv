import time
from typing import Literal
from src.config import WARNING_SECONDS, CRITICAL_SECONDS

class SafetyGuardian:
    def __init__(self):
        """
        Initialize the SafetyGuardian state machine.
        """
        self.last_person_seen_time = time.time()
        self.state = "SAFE"

    def update_status(self, flame_on: bool, person_present: bool) -> Literal["SAFE", "WARNING", "CRITICAL_SHUTOFF"]:
        """
        Update the safety status based on flame and person presence.
        
        Args:
            flame_on: Whether a flame is detected.
            person_present: Whether a person is detected.
            
        Returns:
            Current state: "SAFE", "WARNING", or "CRITICAL_SHUTOFF"
        """
        current_time = time.time()

        if person_present:
            self.last_person_seen_time = current_time
            self.state = "SAFE"
            return "SAFE"

        # If no person is present, check flame status
        if flame_on:
            time_unattended = current_time - self.last_person_seen_time
            
            if time_unattended > CRITICAL_SECONDS:
                self.state = "CRITICAL_SHUTOFF"
            elif time_unattended > WARNING_SECONDS:
                self.state = "WARNING"
            else:
                self.state = "SAFE"
        else:
            # If no flame and no person, it's effectively safe
            self.state = "SAFE"
            # Reset last_person_seen_time to now so that if a flame appears 
            # instantly, we don't immediately error if we haven't seen a person in a while
            self.last_person_seen_time = current_time

        return self.state
