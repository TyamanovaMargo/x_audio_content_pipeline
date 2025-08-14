"""
Snapshot Manager - Track and manage Bright Data snapshots with duplicate prevention
"""

import json
import os
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional

class SnapshotManager:
    def __init__(self, output_dir: str = "output/"):
        self.output_dir = output_dir
        self.snapshots_dir = os.path.join(output_dir, "snapshots")
        self.registry_file = os.path.join(self.snapshots_dir, "snapshot_registry.json")
        
        # Create snapshots directory
        os.makedirs(self.snapshots_dir, exist_ok=True)
        
        # Load existing registry
        self.registry = self._load_registry()
    
    def get_latest_snapshot_for_usernames(self, usernames: List[str]) -> Optional[str]:
        """
        Find the latest snapshot for the same set of usernames to avoid duplicates
        
        Args:
            usernames: List of usernames to check
            
        Returns:
            snapshot_id if found, None otherwise
        """
        usernames_set = set([u.lower().strip() for u in usernames])  # Normalize for comparison
        
        # Sort snapshots by creation time (newest first)
        sorted_snapshots = sorted(
            self.registry.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
        )
        
        for snapshot_id, info in sorted_snapshots:
            existing_usernames = info.get('usernames', [])
            existing_set = set([u.lower().strip() for u in existing_usernames])
            
            # Check if usernames match exactly
            if usernames_set == existing_set:
                print(f"🔍 Found existing snapshot for same usernames: {snapshot_id}")
                print(f"   📅 Created: {info.get('created_at', 'Unknown')}")
                print(f"   📊 Status: {info.get('status', 'Unknown')}")
                print(f"   👥 Accounts: {len(existing_usernames)}")
                return snapshot_id
        
        print(f"🆕 No existing snapshot found for this set of {len(usernames)} usernames")
        return None
    
    def check_snapshot_can_reuse(self, snapshot_id: str) -> bool:
        """
        Check if a snapshot can be reused (completed successfully)
        
        Args:
            snapshot_id: Snapshot ID to check
            
        Returns:
            True if snapshot can be reused, False otherwise
        """
        if snapshot_id not in self.registry:
            return False
            
        info = self.registry[snapshot_id]
        status = info.get('status', 'unknown')
        
        # Can reuse if completed successfully
        if status == 'completed' and info.get('results_count', 0) > 0:
            print(f"✅ Snapshot {snapshot_id} can be reused (status: {status})")
            return True
        elif status in ['running', 'pending']:
            print(f"⏳ Snapshot {snapshot_id} is still processing (status: {status})")
            return True  # Can wait for it to complete
        else:
            print(f"❌ Snapshot {snapshot_id} cannot be reused (status: {status})")
            return False

    def register_snapshot(self, snapshot_id: str, accounts: List[Dict], 
                         trigger_time: str = None) -> Dict:
        """Register a new snapshot with its accounts"""
        
        if not trigger_time:
            trigger_time = datetime.now().isoformat()
        
        # Save accounts that were sent to this snapshot
        accounts_file = os.path.join(self.snapshots_dir, f"{snapshot_id}_accounts.csv")
        pd.DataFrame(accounts).to_csv(accounts_file, index=False)
        
        # Create metadata
        metadata = {
            'snapshot_id': snapshot_id,
            'created_at': trigger_time,
            'status': 'pending',
            'total_accounts': len(accounts),
            'accounts_file': accounts_file,
            'usernames': [acc['username'] for acc in accounts],
            'account_stats': self._calculate_account_stats(accounts)
        }
        
        # Save individual metadata
        metadata_file = os.path.join(self.snapshots_dir, f"{snapshot_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Add to registry
        self.registry[snapshot_id] = metadata
        self._save_registry()
        
        print(f"📝 Snapshot {snapshot_id} registered with {len(accounts)} accounts")
        print(f"📁 Accounts saved: {accounts_file}")
        print(f"📋 Metadata saved: {metadata_file}")
        
        return metadata
    
    def update_snapshot_status(self, snapshot_id: str, status: str, 
                              results_data: List[Dict] = None):
        """Update snapshot status and results"""
        if snapshot_id not in self.registry:
            print(f"⚠️ Snapshot {snapshot_id} not found in registry")
            return
        
        self.registry[snapshot_id]['status'] = status
        self.registry[snapshot_id]['completed_at'] = datetime.now().isoformat()
        
        if results_data:
            self.registry[snapshot_id]['results_count'] = len(results_data)
            self.registry[snapshot_id]['success_rate'] = (
                len(results_data) / self.registry[snapshot_id]['total_accounts'] * 100
            )
            
            # Save results data reference
            results_file = os.path.join(self.snapshots_dir, f"{snapshot_id}_results_sample.json")
            with open(results_file, 'w') as f:
                json.dump(results_data[:5], f, indent=2)  # Save first 5 records as sample
            self.registry[snapshot_id]['results_sample_file'] = results_file
        
        # Update individual metadata file
        metadata_file = os.path.join(self.snapshots_dir, f"{snapshot_id}_metadata.json")
        with open(metadata_file, 'w') as f:
            json.dump(self.registry[snapshot_id], f, indent=2)
        
        self._save_registry()
        print(f"📊 Snapshot {snapshot_id} status updated: {status}")
    
    def get_snapshot_info(self, snapshot_id: str) -> Optional[Dict]:
        """Get information about a specific snapshot"""
        return self.registry.get(snapshot_id)
    
    def list_snapshots(self) -> Dict:
        """List all snapshots"""
        return self.registry
    
    def find_snapshots_by_status(self, status: str) -> List[str]:
        """Find all snapshots with specific status"""
        matching_snapshots = []
        for snapshot_id, info in self.registry.items():
            if info.get('status') == status:
                matching_snapshots.append(snapshot_id)
        return matching_snapshots
    
    def get_reusable_snapshot(self, usernames: List[str]) -> Optional[str]:
        """
        Get a reusable snapshot for the given usernames
        
        Args:
            usernames: List of usernames
            
        Returns:
            snapshot_id if reusable snapshot found, None otherwise
        """
        # First check if we have exact match
        existing_snapshot = self.get_latest_snapshot_for_usernames(usernames)
        
        if existing_snapshot and self.check_snapshot_can_reuse(existing_snapshot):
            return existing_snapshot
        
        return None
    
    def cleanup_old_snapshots(self, keep_days: int = 7):
        """
        Clean up old snapshot files (keep metadata but remove large files)
        
        Args:
            keep_days: Number of days to keep files
        """
        cutoff_date = datetime.now() - pd.Timedelta(days=keep_days)
        cleaned_count = 0
        
        for snapshot_id, info in self.registry.items():
            created_at = info.get('created_at')
            if created_at:
                try:
                    created_date = pd.to_datetime(created_at)
                    if created_date < cutoff_date:
                        # Remove large files but keep metadata
                        accounts_file = info.get('accounts_file')
                        results_file = info.get('results_sample_file')
                        
                        if accounts_file and os.path.exists(accounts_file):
                            os.remove(accounts_file)
                            print(f"🗑️ Removed old accounts file: {accounts_file}")
                            cleaned_count += 1
                        
                        if results_file and os.path.exists(results_file):
                            os.remove(results_file)
                            print(f"🗑️ Removed old results file: {results_file}")
                except Exception as e:
                    print(f"⚠️ Error processing date for {snapshot_id}: {e}")
        
        print(f"🧹 Cleaned up {cleaned_count} old snapshot files")
    
    def print_snapshot_summary(self):
        """Print summary of all snapshots"""
        print("\n📊 SNAPSHOT REGISTRY SUMMARY")
        print("=" * 50)
        
        if not self.registry:
            print("No snapshots found.")
            return
        
        # Group by status
        status_counts = {}
        for info in self.registry.values():
            status = info.get('status', 'unknown')
            status_counts[status] = status_counts.get(status, 0) + 1
        
        print(f"📋 Total snapshots: {len(self.registry)}")
        print(f"📊 Status breakdown:")
        for status, count in status_counts.items():
            print(f"   {status}: {count}")
        
        print(f"\n📝 Recent snapshots:")
        # Show last 5 snapshots
        sorted_snapshots = sorted(
            self.registry.items(),
            key=lambda x: x[1].get('created_at', ''),
            reverse=True
        )[:5]
        
        for snapshot_id, info in sorted_snapshots:
            print(f"\n🆔 {snapshot_id}")
            print(f"   📅 Created: {info.get('created_at', 'Unknown')}")
            print(f"   📊 Status: {info.get('status', 'Unknown')}")
            print(f"   👥 Accounts: {info.get('total_accounts', 0)}")
            
            if info.get('results_count'):
                print(f"   ✅ Results: {info.get('results_count', 0)}")
                print(f"   📈 Success: {info.get('success_rate', 0):.1f}%")
            
            # Show account types
            stats = info.get('account_stats', {})
            if stats:
                print(f"   📋 Account types: {stats}")
    
    def print_duplicate_analysis(self):
        """Analyze and print potential duplicate snapshots"""
        print("\n🔍 DUPLICATE SNAPSHOT ANALYSIS")
        print("=" * 50)
        
        # Group snapshots by usernames
        usernames_groups = {}
        for snapshot_id, info in self.registry.items():
            usernames_key = tuple(sorted(info.get('usernames', [])))
            if usernames_key not in usernames_groups:
                usernames_groups[usernames_key] = []
            usernames_groups[usernames_key].append((snapshot_id, info))
        
        # Find duplicates
        duplicates_found = False
        for usernames_tuple, snapshots in usernames_groups.items():
            if len(snapshots) > 1:
                duplicates_found = True
                print(f"\n📦 Duplicate group ({len(usernames_tuple)} usernames):")
                for snapshot_id, info in snapshots:
                    print(f"   🆔 {snapshot_id}")
                    print(f"      📅 {info.get('created_at', 'Unknown')}")
                    print(f"      📊 {info.get('status', 'Unknown')}")
        
        if not duplicates_found:
            print("✅ No duplicate snapshots found!")
    
    def get_snapshot_accounts(self, snapshot_id: str) -> List[Dict]:
        """
        Get the accounts that were sent to a specific snapshot
        
        Args:
            snapshot_id: Snapshot ID
            
        Returns:
            List of account dictionaries
        """
        if snapshot_id not in self.registry:
            print(f"❌ Snapshot {snapshot_id} not found")
            return []
        
        accounts_file = self.registry[snapshot_id].get('accounts_file')
        if accounts_file and os.path.exists(accounts_file):
            try:
                df = pd.read_csv(accounts_file)
                return df.to_dict('records')
            except Exception as e:
                print(f"❌ Error reading accounts file: {e}")
                return []
        else:
            print(f"❌ Accounts file not found for snapshot {snapshot_id}")
            return []
    
    def export_snapshot_report(self, output_file: str = None):
        """
        Export a comprehensive report of all snapshots
        
        Args:
            output_file: Output file path (default: output/snapshot_report.json)
        """
        if not output_file:
            output_file = os.path.join(self.output_dir, "snapshot_report.json")
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "total_snapshots": len(self.registry),
            "snapshots": self.registry,
            "summary": {
                "status_counts": {},
                "total_accounts_processed": 0,
                "total_results_found": 0
            }
        }
        
        # Calculate summary statistics
        for info in self.registry.values():
            status = info.get('status', 'unknown')
            report["summary"]["status_counts"][status] = report["summary"]["status_counts"].get(status, 0) + 1
            report["summary"]["total_accounts_processed"] += info.get('total_accounts', 0)
            report["summary"]["total_results_found"] += info.get('results_count', 0)
        
        # Save report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"📄 Snapshot report exported: {output_file}")
        return output_file
    
    def _calculate_account_stats(self, accounts: List[Dict]) -> Dict:
        """Calculate statistics about accounts"""
        stats = {}
        for account in accounts:
            status = account.get('status', 'unknown')
            stats[status] = stats.get(status, 0) + 1
        return stats
    
    def _load_registry(self) -> Dict:
        """Load existing registry"""
        if os.path.exists(self.registry_file):
            try:
                with open(self.registry_file, 'r') as f:
                    registry = json.load(f)
                print(f"📚 Loaded {len(registry)} snapshots from registry")
                return registry
            except Exception as e:
                print(f"⚠️ Error loading registry: {e}")
                return {}
        return {}
    
    def _save_registry(self):
        """Save registry to file"""
        try:
            with open(self.registry_file, 'w') as f:
                json.dump(self.registry, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"❌ Error saving registry: {e}")

# Command line interface for testing
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Snapshot Manager CLI")
    parser.add_argument("--output-dir", default="output/", help="Output directory")
    parser.add_argument("--show-summary", action="store_true", help="Show snapshot summary")
    parser.add_argument("--analyze-duplicates", action="store_true", help="Analyze duplicates")
    parser.add_argument("--cleanup", type=int, metavar="DAYS", help="Clean up files older than N days")
    parser.add_argument("--export-report", help="Export report to file")
    parser.add_argument("--get-accounts", metavar="SNAPSHOT_ID", help="Get accounts for snapshot")
    
    args = parser.parse_args()
    
    sm = SnapshotManager(args.output_dir)
    
    if args.show_summary:
        sm.print_snapshot_summary()
    
    if args.analyze_duplicates:
        sm.print_duplicate_analysis()
    
    if args.cleanup:
        sm.cleanup_old_snapshots(args.cleanup)
    
    if args.export_report:
        sm.export_snapshot_report(args.export_report)
    
    if args.get_accounts:
        accounts = sm.get_snapshot_accounts(args.get_accounts)
        print(f"📊 Found {len(accounts)} accounts for snapshot {args.get_accounts}")
        for acc in accounts[:5]:  # Show first 5
            print(f"   - {acc}")
