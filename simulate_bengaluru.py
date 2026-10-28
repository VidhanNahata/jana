import random

class CivicSystemSimulator:
    """
    Mock backend router executing the absolute end-to-end integration natively via isolated modules.
    Tests physical Ingestion -> PostGIS routing -> NLP merge -> Dispatch.
    """
    def __init__(self):
        self.master_db = []
        
    def ingest_payload(self, user_id, image_metadata, text_payload, lat, lng):
        print(f"\n[API INGESTION] Payload submitted by {user_id} at {lat}, {lng}")
        
        # 1. Verification Simulation (Image Forensic Firewall)
        compression = image_metadata.get('jpeg_compression', 95)
        ela_noise = image_metadata.get('ela_suspicion_score', 0.1)
        
        if compression < 20 or ela_noise > 0.8:
             print(f" [X] REJECTED: Convolutional ELA Failure. Severe Forgery Detected. User {user_id} flagged.")
             return {"status": "Terminal Rejection"}
        print(" [✓] ELA Verification Check Passed")
             
        # 2. NLP Deduplication Simulation (IndicSBERT + SRID 3395 Mocking)
        is_duplicate = False
        target_ticket = None
        for master in self.master_db:
            # Physical mock: If distance is ~100m (0.001 degree approximation limit)
            if abs(master['lat'] - lat) < 0.0015 and master['category'] == text_payload['category']:
                is_duplicate = True
                target_ticket = master
                break
                
        if is_duplicate:
             print(f" [✓] IndicSBERT Match (>0.90 Cosine). Merged duplicate into Master Ticket #{target_ticket['id']}")
             target_ticket['compound_reports'] += 1
        else:
             target_ticket = {
                 "id": f"BLR-{random.randint(1000,9999)}",
                 "lat": lat, "lng": lng,
                 "category": text_payload['category'],
                 "compound_reports": 1
             }
             self.master_db.append(target_ticket)
             print(f" [✓] Novel geographic constraint. Created new Ticket #{target_ticket['id']}")

        # 3. ITIL / Random Forest Priority Loop
        urgency = "High" if target_ticket['compound_reports'] >= 2 else "Low"
        severity = text_payload.get('severity', 'Medium')
        
        priority = "P4" 
        if urgency == "High" and severity == "High": priority = "P1"
        elif urgency == "High" or severity == "High": priority = "P2"
        elif severity == "Medium": priority = "P3"
        print(f" [✓] ML Matrix computed Threat Vector -> Priority Assigned: {priority}")
        
        # 4. Department Route Mapping (Bengaluru specific structural bounds)
        bbmp_routes = {
            "Roads": "BBMP PWD Sub-Div", 
            "Water": "BWSSB Emergency Layer", 
            "Electricity": "BESCOM Outage Desk"
        }
        department = bbmp_routes.get(text_payload['category'], "BBMP Gen-Admin")
        
        # 5. SLA Rule Binding
        sla = "1 Hour" if priority == "P1" else ("24 Hours" if priority == "P2" else "3 Days")
        print(f" [✓] REST ROUTER: Fired webhook to {department} (SLA Dispatch Limit: {sla})")
        
        return {
            "master_id": target_ticket['id'], 
            "status": "Live Tracking", 
            "department": department, 
            "sla": sla
        }

if __name__ == "__main__":
    print("=== END TO END PHYSICAL BENGALURU INTEGRATION SIMULATION ===")
    sim = CivicSystemSimulator()
    
    # User 1: Posts valid pothole
    print("\n--- Event 1: User Alpha (Koramangala Block 5) ---")
    sim.ingest_payload(
        user_id="USR-901",
        image_metadata={"jpeg_compression": 85, "ela_suspicion_score": 0.05}, 
        text_payload={"category": "Roads", "severity": "Medium", "text": "Pothole near Forum Mall stopping traffic."},
        lat=12.9352, lng=77.6145
    )
    
    # User 2: 30 minutes later, someone nearby uploads Hindi duplicate
    print("\n--- Event 2: User Beta (Koramangala Block 5) ---")
    sim.ingest_payload(
        user_id="USR-114",
        image_metadata={"jpeg_compression": 98, "ela_suspicion_score": 0.02}, 
        text_payload={"category": "Roads", "severity": "High", "text": "सड़क पर बहुत बड़ा गड्ढा है"},
        lat=12.9353, lng=77.6146
    )
    
    # User 3: Angry citizen tries to upload a photoshopped image of a water explosion
    print("\n--- Event 3: User Charlie (Majestic Station) ---")
    sim.ingest_payload(
        user_id="USR-999 (MALICIOUS)",
        image_metadata={"jpeg_compression": 15, "ela_suspicion_score": 0.94}, # Intense Splice Signature
        text_payload={"category": "Water", "severity": "High"},
        lat=12.9716, lng=77.5946
    )
