import asyncio
import csv
import json
import os
import glob
import random
import re
import signal
import logging
from datetime import datetime, timedelta
from collections import Counter
from dataclasses import dataclass, field
from typing import (
    List, Dict, Set, Optional, Tuple, Any, Final, Callable, ClassVar
)
from textblob import TextBlob
from twikit import Client

# =============================================================================
# REGION: EXCEPTIONS & CONSTANTS
# =============================================================================

class ScraperError(Exception): """Base exception for Twitter scraping."""
class AuthError(ScraperError): """Raised when authentication fails."""
class RateLimitError(ScraperError): """Raised specifically for 429 rate limits."""
class ExtractionError(ScraperError): """Raised when parsing tweet data fails."""

class Theme:
    """Centralized ANSI styling constants for the Dashboard interface."""
    RESET: Final[str] = "\033[0m"
    BOLD: Final[str] = "\033[1m"
    RED: Final[str] = "\033[0;31m"
    GREEN: Final[str] = "\033[0;32m"
    YELLOW: Final[str] = "\033[0;33m"
    BLUE: Final[str] = "\033[0;34m"
    CYAN: Final[str] = "\033[0;36m"
    MAGENTA: Final[str] = "\033[0;35m"

@dataclass(frozen=True)
class AppConfig:
    """Core framework configuration and parameters."""
    FIELDNAMES: ClassVar[List[str]] = [
        "keyword", "tweet_id", "username", "handle", "text", "sentiment", 
        "url", "datetime", "likes", "retweets", "replies", "tweets_per_date"
    ]
    COOLDOWN_TIERS: ClassVar[List[int]] = [60, 300, 1800, 3600] # 1m, 5m, 30m, 1h
    MAX_ROTATION_FAILS: ClassVar[int] = 3
    PAGINATION_SLEEP: ClassVar[float] = 1.6
    DASHBOARD_WIDTH: ClassVar[int] = 76

# =============================================================================
# REGION: CORE INITIALIZATION
# =============================================================================

class AppState:
    STOP_REQUESTED = False

def _interupt_handler(signum, frame):
    if AppState.STOP_REQUESTED:
        print("\n\033[0;31m[FATAL]\033[0m Forced Exit.")
        os._exit(1)
    print("\n\033[0;33m[WAIT]\033[0m Interrupt detected! Returning to menu... (Press Ctrl+C again to force exit)")
    AppState.STOP_REQUESTED = True
    raise KeyboardInterrupt

signal.signal(signal.SIGINT, _interupt_handler)

def init_twikit_patch():
    """Applies a comprehensive Super Patch to twikit to bypass recent breaking Twitter API changes."""
    try:
        import twikit.x_client_transaction.transaction as _tx_mod
        from twikit.utils import find_dict, Result
        from twikit.tweet import Tweet
        from functools import partial
        import twikit.client.gql as _gql_mod
        from twikit.client.gql import Endpoint

        # [1] Fix: Official PR #410 On-Demand JS Discovery Logic
        _tx_mod.ON_DEMAND_FILE_REGEX = re.compile(r""",(\d+):["']ondemand\.s["']""", flags=(re.VERBOSE | re.MULTILINE))
        _tx_mod.ON_DEMAND_HASH_PATTERN = r',{}:"([0-9a-f]+)"'
        _tx_mod.INDICES_REGEX = re.compile(r"\[(\d+)\],\s*16")

        async def _patched_get_indices(self, home_page_response, session, headers):
            response = self.validate_response(home_page_response) or self.home_page_response
            resp_str = str(response)
            
            match = _tx_mod.ON_DEMAND_FILE_REGEX.search(resp_str)
            if not match:
                match = re.search(r'["\']ondemand\.s["\']:\s*["\'](\w+)["\']', resp_str)
                if not match: raise Exception("ON_DEMAND_FILE_REGEX not found")
                filename = match.group(1)
            else:
                on_demand_idx = match.group(1)
                hash_regex = re.compile(_tx_mod.ON_DEMAND_HASH_PATTERN.format(on_demand_idx))
                match = hash_regex.search(resp_str)
                if not match: raise Exception("ON_DEMAND_HASH_PATTERN not found")
                filename = match.group(1)

            url = f"https://abs.twimg.com/responsive-web/client-web/ondemand.s.{filename}a.js"
            resp = await session.request(method="GET", url=url, headers=headers)
            
            indices_match = _tx_mod.INDICES_REGEX.finditer(str(resp.text))
            key_byte_indices = [item.group(1) for item in indices_match] # Group 1 as per PR #410
            
            if not key_byte_indices:
                matches = re.findall(r'\((\w)\[(\d+)\],\s*16\)', str(resp.text))
                key_byte_indices = [m[1] for m in matches]
                
            if not key_byte_indices: raise Exception("Couldn't get KEY_BYTE indices")
            return int(key_byte_indices[0]), list(map(int, key_byte_indices[1:]))

        _tx_mod.ClientTransaction.get_indices = _patched_get_indices

        # [2] Fix: Update SearchTimeline GraphQL Endpoint and Features (Latest: R0u1RWRf748KzyGBXvOYRA)
        _gql_mod.Endpoint.SEARCH_TIMELINE = _gql_mod.Endpoint.url('R0u1RWRf748KzyGBXvOYRA/SearchTimeline')
        
        # Override gql_get to inject latest features if it's a search request
        original_gql_get = _gql_mod.GQLClient.gql_get
        async def _patched_gql_get(self, endpoint, variables, features=None, headers=None, extra_params=None, **kwargs):
            try:
                response_data = await original_gql_get(self, endpoint, variables, features, headers, extra_params, **kwargs)
            except Exception as e:
                if '404' in str(e) and 'SearchTimeline' in endpoint:
                    # Multi-lane Fallback: Cycle through known working IDs
                    alt_ids = ['flaR-PUMshxFWZWPNpq4zA', 'nK1dw4oV3_S97puv7wOshA', 't-97775-SearchTimeline']
                    current_id = endpoint.split('/')[-2]
                    
                    response_data = None
                    for alt_id in alt_ids:
                        if alt_id == current_id: continue
                        alt_endpoint = endpoint.replace(current_id, alt_id)
                        try:
                            print(f"[{Theme.YELLOW}Note{Theme.RESET}] Search ID {current_id} failed, trying {alt_id}...")
                            response_data = await original_gql_get(self, alt_endpoint, variables, features, headers, extra_params, **kwargs)
                            if response_data: break
                        except: continue
                    
                    if not response_data: raise
                else: raise
            
            # Universal JSON Extractor
            try:
                if not isinstance(response_data, dict):
                    raw_txt = ""
                    if hasattr(response_data, 'text'):
                        raw_txt = response_data.text.strip()
                    else:
                        raw_txt = str(response_data).strip()
                    
                    # Regex to find the outermost { ... }
                    import re
                    match = re.search(r'(\{.*\})', raw_txt, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        try:
                            response_data = json.loads(json_str)
                        except:
                            # If direct JSON fails, try cleaning single quotes (JS-style)
                            try:
                                import ast
                                response_data = ast.literal_eval(json_str)
                            except: pass
            except Exception as e:
                print(f"[{Theme.YELLOW}Trace{Theme.RESET}] Extractor failed: {e}")

            # Ensure we don't return a raw object to a function expecting JSON
            if not isinstance(response_data, dict):
                raw_val = response_data.text if hasattr(response_data, 'text') else str(response_data)
                response_data = {'errors': [{'message': 'Raw response failed to parse as JSON', 'code': 0}], 'raw': raw_val}
            else:
                # Success Log (Silent if everything is working)
                pass

            # Black Box Logger
            if 'SearchTimeline' in endpoint:
                try:
                    os.makedirs('scratch', exist_ok=True)
                    with open('scratch/last_gql_response.json', 'w') as f:
                        json.dump(response_data, f, indent=2, default=str)
                except: pass

            # API Error Reporting
            errors = response_data.get('errors')
            if errors and isinstance(errors, list):
                first_error = errors[0] if isinstance(errors[0], dict) else {}
                error_code = first_error.get('code')
                error_message = first_error.get('message')
                if error_code or error_message:
                    print(f"[{Theme.RED}API Error {error_code}{Theme.RESET}] {error_message}")

            return response_data
        
        _gql_mod.GQLClient.gql_get = _patched_gql_get

        # [3] Fix: Search Result Parser (handles Twitter's new user data layout)
        from twikit.client.client import Client, tweet_from_data
        from twikit.user import User

        def _patch_user_data(user_data):
            """Twitter moved name/screen_name/created_at from user.legacy to user.core.
            Twikit's User() reads from legacy, so we copy the fields back."""
            if not isinstance(user_data, dict) or 'legacy' not in user_data:
                return
            legacy = user_data['legacy']
            core = user_data.get('core', {})
            # Migrate core fields into legacy so twikit's User() can read them
            for field in ('name', 'screen_name', 'created_at'):
                if field not in legacy and field in core:
                    legacy[field] = core[field]
            # Ensure rest_id exists
            if 'rest_id' not in user_data:
                user_data['rest_id'] = legacy.get('id_str', '0')
            # Backfill defaults for fields User() expects but may be missing
            defaults = {
                'profile_image_url_https': '', 'location': '', 'description': '',
                'entities': {'description': {'urls': []}}, 'pinned_tweet_ids_str': [],
                'verified': False, 'possibly_sensitive': False, 'can_dm': False,
                'can_media_tag': False, 'want_retweets': False, 'statuses_count': 0,
                'translator_type': '', 'withheld_in_countries': [],
            }
            for k, v in defaults.items():
                if k not in legacy:
                    legacy[k] = v

        def _safe_user(client, user_data):
            """Build a User object safely, falling back to a stub on failure."""
            if not user_data or not isinstance(user_data, dict):
                return _stub_user()
            try:
                _patch_user_data(user_data)
                return User(client, user_data)
            except Exception:
                legacy = user_data.get('legacy', {})
                core = user_data.get('core', {})
                return type('StubUser', (), {
                    'name': core.get('name') or legacy.get('name', 'Unknown'),
                    'screen_name': core.get('screen_name') or legacy.get('screen_name', 'unknown'),
                    'rest_id': user_data.get('rest_id', '0'), 'id': user_data.get('rest_id', '0'),
                })()

        def _stub_user():
            return type('StubUser', (), {
                'name': 'Unknown', 'screen_name': 'unknown', 'rest_id': '0', 'id': '0',
            })()

        async def _patched_search_tweet(self, query, product, count=20, cursor=None):
            product = product.capitalize()
            variables = {
                'rawQuery': query, 'count': count,
                'querySource': 'typed_query', 'product': product,
                'withGrokTranslatedBio': True,
            }
            if cursor:
                variables['cursor'] = cursor

            # Feature flags from PR #419
            features = {
                'rweb_video_screen_enabled': False,
                'rweb_cashtags_enabled': True,
                'profile_label_improvements_pcf_label_in_post_enabled': True,
                'responsive_web_graphql_timeline_navigation_enabled': True,
                'responsive_web_graphql_skip_user_profile_image_extensions_enabled': False,
                'creator_subscriptions_tweet_preview_api_enabled': True,
                'communities_web_enable_tweet_community_results_fetch': True,
                'c9s_tweet_anatomy_moderator_badge_enabled': True,
                'responsive_web_grok_analyze_button_fetch_trends_enabled': False,
                'responsive_web_grok_analyze_post_followups_enabled': True,
                'responsive_web_jetfuel_frame': True,
                'responsive_web_grok_share_attachment_enabled': True,
                'responsive_web_grok_annotations_enabled': True,
                'articles_preview_enabled': True,
                'responsive_web_edit_tweet_api_enabled': True,
                'graphql_is_translatable_rweb_tweet_is_translatable_enabled': True,
                'view_counts_everywhere_api_enabled': True,
                'longform_notetweets_consumption_enabled': True,
                'responsive_web_twitter_article_tweet_consumption_enabled': True,
                'responsive_web_grok_show_grok_translated_post': True,
                'responsive_web_grok_analysis_button_from_backend': True,
                'responsive_web_grok_image_annotation_enabled': True,
                'responsive_web_grok_imagine_annotation_enabled': True,
                'responsive_web_grok_community_note_auto_translation_is_enabled': True,
                'responsive_web_enhance_cards_enabled': False,
                'responsive_web_search_grid_enabled': True,
                'freedom_of_speech_not_reach_fetch_enabled': True,
                'standardized_nudges_misinfo': True,
                'tweet_with_visibility_results_prefer_gql_limited_actions_policy_enabled': True,
                'longform_notetweets_rich_text_read_enabled': True,
                'longform_notetweets_inline_media_enabled': True,
                'post_ctas_fetch_enabled': True,
                'content_disclosure_indicator_enabled': True,
                'content_disclosure_ai_generated_indicator_enabled': True,
                'premium_content_api_read_enabled': False,
                'responsive_web_profile_redirect_enabled': False,
                'rweb_tipjar_consumption_enabled': False,
                'verified_phone_label_enabled': False,
            }

            response = await self.gql.gql_get(
                _gql_mod.Endpoint.SEARCH_TIMELINE, variables, features
            )

            # --- Parse the GQL response ---
            results = []
            next_cursor = None
            previous_cursor = None

            # Navigate to the entries list via the standard GQL path
            entries = []
            try:
                instructions = response['data']['search_by_raw_query']['search_timeline']['timeline']['instructions']
                for inst in instructions:
                    entries.extend(inst.get('entries', []))
            except (KeyError, TypeError):
                # Fallback: recursively find any list of entries
                found = find_dict(response, 'entries', find_one=True)
                if found:
                    entries = found[0] if isinstance(found[0], list) else []

            for entry in entries:
                eid = entry.get('entryId', '')

                # --- Cursor extraction ---
                if 'cursor-bottom' in eid:
                    next_cursor = entry.get('content', {}).get('value')
                    continue
                elif 'cursor-top' in eid:
                    previous_cursor = entry.get('content', {}).get('value')
                    continue

                # --- Tweet extraction ---
                if not eid.startswith('tweet-'):
                    continue

                try:
                    # Standard path: content.itemContent.tweet_results.result
                    item_content = entry['content']['itemContent']
                    tweet_result = item_content['tweet_results']['result']

                    # Handle "tweet" wrapper (e.g. visibility-filtered tweets)
                    if 'tweet' in tweet_result:
                        tweet_result = tweet_result['tweet']

                    # Ensure rest_id
                    if 'rest_id' not in tweet_result and 'legacy' in tweet_result:
                        tweet_result['rest_id'] = tweet_result['legacy'].get('id_str')

                    # Build User from core.user_results.result
                    user_data = tweet_result.get('core', {}).get('user_results', {}).get('result')
                    user = _safe_user(self, user_data)

                    tweet = Tweet(self, tweet_result, user)
                    results.append(tweet)
                except Exception:
                    pass

            print(f"[{Theme.CYAN}Debug{Theme.RESET}] Parsed {len(results)} tweets from {len(entries)} entries.")

            return Result(
                results,
                partial(self.search_tweet, query, product, count, next_cursor),
                next_cursor,
                partial(self.search_tweet, query, product, count, previous_cursor),
                previous_cursor
            )

        Client.search_tweet = _patched_search_tweet
        
    except Exception as e:
        print(f"[{Theme.RED}Warning{Theme.RESET}] Twikit Super Patch failed: {e}")

init_twikit_patch()

# =============================================================================
# REGION: UI & LOGGING FRAMEWORK
# =============================================================================

class DashboardLogger(logging.Formatter):
    """Custom color-coded formatter. Integrates seamlessly with Theme logic."""
    def format(self, record):
        cat = record.levelname
        colors = {
            "INFO": Theme.CYAN, "SUCCESS": Theme.GREEN, "LIMIT": Theme.YELLOW + Theme.BOLD, 
            "ERROR": Theme.RED, "FATAL": Theme.RED + Theme.BOLD, "AUTH": Theme.MAGENTA,
            "TASK": Theme.BLUE + Theme.BOLD, "FILE": Theme.BLUE, "WAIT": Theme.YELLOW
        }
        color = colors.get(cat, Theme.RESET)
        return f" {color}{Theme.BOLD}{cat:<8}{Theme.RESET} │ {record.getMessage()}"

# Register Custom Logging levels
for lvl, name in [(25, "SUCCESS"), (35, "LIMIT"), (15, "AUTH"), (16, "TASK"), (17, "FILE"), (18, "WAIT")]:
    logging.addLevelName(lvl, name)

logger = logging.getLogger("TwitterScraper")
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(DashboardLogger())
logger.addHandler(sh)

def log(msg: str, cat: str = "INFO"):
    mappings = {"INFO": 20, "SUCCESS": 25, "LIMIT": 35, "ERROR": 40, "FATAL": 50, "AUTH": 15, "TASK": 16, "FILE": 17, "WAIT": 18}
    logger.log(mappings.get(cat, 20), msg)

def print_header(title: str):
    """Visual framed header, adjusting dynamically to AppConfig width."""
    w = AppConfig.DASHBOARD_WIDTH
    print(f"\n{Theme.BLUE}{'═'*w}{Theme.RESET}\n{Theme.BOLD}{Theme.CYAN}  {title:^{w-4}}  {Theme.RESET}\n{Theme.BLUE}{'═'*w}{Theme.RESET}")

def get_input(prompt: str, default: Any = None, type_func: Callable = str) -> Any:
    """Interactive CLI prompt with explicit default handling and validation."""
    p = f"{Theme.BOLD}{Theme.CYAN}➤ {prompt}{Theme.RESET}" + (f" {Theme.YELLOW}({default}){Theme.RESET}: " if default is not None else ": ")
    val = input(p).strip()
    if not val: return default
    try: return type_func(val)
    except ValueError:
        log(f"Invalid input type. Using default: {default}", "ERROR")
        return default

# =============================================================================
# REGION: DATA PROCESSING LAYER (O(1) Memory Engine)
# =============================================================================

class DataEngine:
    """Manages high-performance CSV I/O, optimized deduplication, and NLP sentiment."""
    
    @staticmethod
    def _path(p: str) -> str:
        return p if p.lower().endswith('.csv') else p + '.csv'

    @staticmethod
    def analyze_sentiment(text: str) -> str:
        """Returns a stable, human-readable sentiment classification."""
        if not text: return "Neutral"
        try:
            pol = TextBlob(text).sentiment.polarity
            if pol > 0.1: return "Positive"
            if pol < -0.1: return "Negative"
            return "Neutral"
        except: return "Neutral"

    @classmethod
    def sync_to_disk(cls, data: List[Dict], path: str):
        """Atomic write protocol. Aggregates statistical columns before flush."""
        if not data: return
        path = cls._path(path)
        
        # Date Normalization & Frequency Mapping
        date_map = Counter()
        for row in data:
            dt_raw = str(row.get('datetime', ''))
            for fmt in ('%a %b %d %H:%M:%S %z %Y', '%Y-%m-%d %H:%M:%S', '%Y-%m-%d'):
                try:
                    obj = datetime.strptime(dt_raw, fmt)
                    row['datetime'] = obj.strftime('%Y-%m-%d %H:%M:%S')
                    row['_date_key'] = obj.strftime('%Y-%m-%d')
                    break
                except ValueError: continue
            if '_date_key' not in row: row['_date_key'] = dt_raw[:10]
            date_map[row['_date_key']] += 1
            
        for row in data:
            row['tweets_per_date'] = date_map.get(row.pop('_date_key', ''), 0)

        try:
            with open(path, 'w', newline='', encoding='utf-8') as fs:
                wr = csv.DictWriter(fs, fieldnames=AppConfig.FIELDNAMES, extrasaction='ignore')
                wr.writeheader()
                wr.writerows(data)
            log(f"Sync complete: {len(data)} records secured in '{os.path.basename(path)}'", "FILE")
        except Exception as e: log(f"Critical Sync Failure: {e}", "FATAL")

    @classmethod
    def extract_signatures(cls, path: str) -> Tuple[Set[str], Set[str]]:
        """Loads ID and Text signatures. Preserves RAM during massive scrapes."""
        path = cls._path(path)
        ids, texts = set(), set()
        if not os.path.exists(path): return ids, texts
        
        try:
            with open(path, 'r', encoding='utf-8') as fs:
                for row in csv.DictReader(fs):
                    if 'tweet_id' in row: ids.add(str(row['tweet_id']))
                    if 'text' in row: texts.add(row['text'].strip())
        except Exception as e:
            log(f"Signature Extraction failed: {e}", "ERROR")
        return ids, texts

    @classmethod
    def load_full(cls, path: str) -> List[Dict]:
        """Loads complete historical dataset when memory pressure is acceptable."""
        path = cls._path(path)
        if not os.path.exists(path): return []
        try:
            with open(path, 'r', encoding='utf-8') as fs: return list(csv.DictReader(fs))
        except: return []

# =============================================================================
# REGION: ACCOUNT INFRASTRUCTURE
# =============================================================================

class TwitterAccount:
    """State-aware representation of a tracked X session."""
    __slots__ = ('idx', 'cookies_file', 'client', 'fails', 'resume_until', 'is_busy')

    def __init__(self, idx: int, file: str):
        self.idx = idx + 1
        self.cookies_file = file
        self.client = Client('en-US')
        self.fails = 0
        self.resume_until: Optional[datetime] = None
        self.is_busy = False

    async def __aenter__(self):
        self.is_busy = True; return self

    async def __aexit__(self, *exc):
        self.is_busy = False

    async def auth(self) -> bool:
        """Handles automated JSON cookie loading or manual extraction flows."""
        prev_auth = None
        if os.path.exists(self.cookies_file):
            try:
                with open(self.cookies_file, 'r') as f:
                    c_data = json.load(f)
                    if isinstance(c_data, dict) and 'auth_token' in c_data:
                        prev_auth = c_data['auth_token']
            except: pass
            
            log(f"Acct {self.idx}: Validating saved session...", "AUTH")
            while True:
                try:
                    # Reinitialize the client each attempt to discard any stale
                    # HTTP session / transaction state from the previous failure.
                    self.client = Client('en-US')
                    self.client.load_cookies(self.cookies_file)
                    log(f"Acct {self.idx}: Testing API connection...", "WAIT")
                    await self.client.search_tweet('test', 'Latest', count=1)
                    log(f"Acct {self.idx}: ACTIVE (API OK)", "SUCCESS")
                    return True
                except Exception as e:
                    log(f"Acct {self.idx}: INACTIVE or INVALID ({e}).", "ERROR")

                    print(f"\n  {Theme.BOLD}{Theme.YELLOW}OPTIONS:{Theme.RESET}"
                          f"\n  {Theme.YELLOW}(r){Theme.RESET} Retry  — update the cookies file on disk first, then press r to re-test"
                          f"\n  {Theme.YELLOW}(y){Theme.RESET} Re-auth — enter credentials or paste cookies manually"
                          f"\n  {Theme.YELLOW}(n){Theme.RESET} Skip   — exclude this account from the session\n")

                    ans = get_input(f"Account {self.idx}: choice (r/y/n)", "y")
                    ans_lower = str(ans).lower()
                    if ans_lower == 'r':
                        log(f"Waiting for you to update '{os.path.basename(self.cookies_file)}'...", "WAIT")
                        print(f"  {Theme.CYAN}How to get fresh cookies:{Theme.RESET}"
                              f"\n    1. Open Twitter/X in your browser and log in"
                              f"\n    2. DevTools (F12) → Application → Cookies → x.com"
                              f"\n    3. Copy the values of  auth_token  and  ct0"
                              f"\n    4. Update the file:  {self.cookies_file}"
                              f"\n    5. Press Enter below when done\n")
                        input(f"{Theme.BOLD}  ↵ Press Enter to retry the connection test...{Theme.RESET}")
                        # Loop continues → load_cookies picks up the updated file
                        continue
                    elif ans_lower != 'y':
                        log(f"Skipping Account {self.idx}.", "WAIT")
                        return False
                    break

        print("\n" + "-" * AppConfig.DASHBOARD_WIDTH)
        print(f"{Theme.BOLD}{Theme.CYAN}SECURITY ALERT: Authentication Required | Account {self.idx}{Theme.RESET}")
        if prev_auth:
            masked = f"{prev_auth[:8]}...{prev_auth[-8:]}" if len(prev_auth) > 16 else prev_auth
            print(f"  {Theme.YELLOW}Previous auth_token: {masked}{Theme.RESET}")
        print("Please enter your X credentials to inject standard cookies. (Press Enter on Username to skip)")
        print("-" * AppConfig.DASHBOARD_WIDTH)
        
        try:
            u = input(f"{Theme.BOLD}? Username (Handle): {Theme.RESET}").strip()
            if not u:
                log(f"Skipping Account {self.idx}.", "WAIT")
                return False
            e = input(f"{Theme.BOLD}? Email: {Theme.RESET}")
            p = input(f"{Theme.BOLD}? Password: {Theme.RESET}")
            
            log("Handshaking with X servers...", "AUTH")
            await self.client.login(auth_info_1=u, auth_info_2=e, password=p)
            self.client.save_cookies(self.cookies_file)
            log("Authentication Successful!", "SUCCESS")
            return True
        except Exception as err:
            err_msg = str(err)
            if "status: 403" in err_msg or "cloudflare" in err_msg.lower():
                log("Cloudflare firewall detected. Reverting to manual cookie injection.", "FATAL")
            else:
                log(f"Automated login failed: {err_msg}", "ERROR")
            
            print("\n  " + Theme.BOLD + "MANUAL COOKIE INJECTION PROTOCOL (Press Enter on auth_token to skip):" + Theme.RESET)
            print("  1. Login to Twitter.com in your browser")
            print("  2. Open DevTools (F12) -> Application Tab -> Cookies")
            at = input(f"{Theme.BOLD}  ? Paste 'auth_token': {Theme.RESET}").strip()
            if not at:
                log(f"Skipping Account {self.idx}.", "WAIT")
                return False
            ct = input(f"{Theme.BOLD}  ? Paste 'ct0': {Theme.RESET}").strip()
            if at and ct:
                with open(self.cookies_file, 'w') as fs: json.dump({'auth_token': at, 'ct0': ct}, fs)
                try:
                    self.client.load_cookies(self.cookies_file)
                    await self.client.search_tweet('test', 'Latest', count=1)
                    log("Cookies hardcoded and validated. ACTIVE.", "SUCCESS")
                    return True
                except Exception as e:
                    log(f"Provided cookies failed validation: {e}", "ERROR")
        return False

    def trigger_cooldown(self, is_429: bool = False):
        """Tiered penalization system for accounts reporting network API stress."""
        self.fails += 1
        if is_429:
            sec = 900 if self.fails == 1 else (1800 if self.fails == 2 else 3600)
            reason = "Rate Limit Strike"
        else:
            sec = AppConfig.COOLDOWN_TIERS[min(self.fails-1, len(AppConfig.COOLDOWN_TIERS)-1)]
            reason = "Network Timeout"
        
        self.resume_until = datetime.now() + timedelta(seconds=sec)
        log(f"Acct {self.idx} ({reason}): Imposed {sec//60}m penalty. Ready at {self.resume_until.strftime('%H:%M:%S')}", "LIMIT")

    def available(self) -> bool:
        """Determines if the account is healthy and unoccupied."""
        return not self.is_busy and (not self.resume_until or self.resume_until < datetime.now())

class WorkerPool:
    """Load balancer for the active TwitterAccount fleet."""
    def __init__(self, accs: List[TwitterAccount]): self.accs = accs
    
    def lease_best(self, start_idx: int = 0) -> Tuple[Optional[TwitterAccount], int]:
        """Cyclical scanning for the first healthy node."""
        total = len(self.accs)
        for i in range(total):
            idx = (start_idx + i) % total
            acc = self.accs[idx]
            if acc.available(): return acc, idx
        return None, -1

    async def stall_for_health(self):
        """Intelligent sleep state when the entire pool is penalized."""
        now = datetime.now()
        waits = [int((a.resume_until - now).total_seconds()) for a in self.accs if a.resume_until and a.resume_until > now]
        if waits:
            d = min(waits) + 2
            log(f"Global Pool Exhaustion. Forcing system stall for {d} seconds...", "WAIT")
            await asyncio.sleep(d)

# =============================================================================
# REGION: SCRAPING LOGIC ENGINE
# =============================================================================

@dataclass
class ScrapeTask:
    """Standardized manifest for executing a data hook."""
    __slots__ = ('query', 'limit', 'ids', 'texts', 'since', 'until', 'max_dup')
    query: str; limit: int; ids: Set[str]; texts: Set[str]
    since: Optional[str]; until: Optional[str]; max_dup: int

class ScrapingEngine:
    """Universal controller for executing, tracking, and validating API loops."""
    
    @classmethod
    async def process(cls, pool: WorkerPool, pref_idx: int, t: ScrapeTask) -> Tuple[List[Dict], int]:
        """The 'Safety Fuse' algorithm. Handles rotation, dups, and network faults."""
        extracted, n, consecutive_dups, rotation_fails = [], 0, 0, 0
        duplicates_skipped = 0
        q_final = f"{t.query}{' since:'+t.since if t.since else ''}{' until:'+t.until if t.until else ''}"
        
        curr_idx = pref_idx
        try:
            while len(extracted) < t.limit:
                if AppState.STOP_REQUESTED:
                    log("Interrupt flag active. Concluding search loop prematurely.", "WAIT")
                    break
                acc, active_idx = pool.lease_best(curr_idx); curr_idx = active_idx
                if not acc: 
                    await pool.stall_for_health(); continue

                async with acc:
                    log(f"Engine [Node {acc.idx}] ➔ Active Profile: '{t.query}'", "INFO")
                    try:
                        try: tws = await acc.client.search_tweet(q_final, 'Latest')
                        except Exception as e:
                            if '404' in str(e): 
                                log("No recent data. Shifting protocol to 'Top' tweets...", "WAIT")
                                tws = await acc.client.search_tweet(q_final, 'Top')
                            else: raise e

                        while len(extracted) < t.limit and tws:
                            found_fresh = False
                            for tweet in tws:
                                # Stop processing the page once the limit is already satisfied
                                if len(extracted) >= t.limit:
                                    break

                                msg = tweet.text.replace('\n', ' ')
                                t_id = str(tweet.id)
                                if t_id in t.ids or msg.strip() in t.texts:
                                    log(f"Skip: @{tweet.user.screen_name:<16} │ Duplicate Entity", "FILE")
                                    duplicates_skipped += 1
                                    continue

                                # Append first, then register — so the ID is only marked
                                # as seen once the tweet is safely stored.
                                extracted.append({
                                    "keyword": t.query, "tweet_id": tweet.id, "username": tweet.user.name, "handle": tweet.user.screen_name,
                                    "text": msg, "sentiment": DataEngine.analyze_sentiment(msg), "url": f"https://x.com/status/{tweet.id}",
                                    "datetime": tweet.created_at, "likes": tweet.favorite_count, "retweets": tweet.retweet_count, "replies": tweet.reply_count
                                })
                                t.ids.add(t_id)
                                t.texts.add(msg.strip())
                                n += 1
                                found_fresh = True
                                log(f"+ Vault: @{tweet.user.screen_name:<15} │ Trajectory: {n}/{t.limit} │ Extracted: {len(extracted)}", "SUCCESS")

                            if not found_fresh:
                                consecutive_dups += 1
                                if consecutive_dups >= t.max_dup: 
                                    log(f"Search Saturation reached: {consecutive_dups} consecutive duplicate pages. Concluding segment.", "WAIT")
                                    break
                            else:
                                consecutive_dups = 0; acc.fails = 0 # Reward healthy progress
                            
                            if len(extracted) < t.limit:
                                if AppState.STOP_REQUESTED: break
                                await asyncio.sleep(AppConfig.PAGINATION_SLEEP)
                                tws = await tws.next()
                        break
                        
                    except Exception as ex:
                        if '404' in str(ex):
                            log("Search index empty or unavailable (404). Concluding segment.", "WAIT")
                            break
                            
                        acc.trigger_cooldown(is_429=('429' in str(ex)))
                        log(f"Connection Fault [Node {acc.idx}]: {ex}", "ERROR")
                        curr_idx = (curr_idx + 1) % len(pool.accs)
                        rotation_fails += 1
                        
                        if rotation_fails >= len(pool.accs) * AppConfig.MAX_ROTATION_FAILS:
                            log("Critical Fuse Tripped! Exceeded maximum safe rotation faults. Aborting.", "FATAL")
                            break
                        continue
        except KeyboardInterrupt:
            log(f"Manual Override (Ctrl+C). Preserving {len(extracted)} securely extracted payloads...", "WAIT")

        log(f"Segment Complete: {len(extracted)} payloads extracted │ {duplicates_skipped} duplicates bypassed.", "SUCCESS")
        return extracted, n

# =============================================================================
# REGION: ANALYTICA DASHBOARD (v2.0 UI)
# =============================================================================

class AnalyticaDashboard:
    """Interactive Control Layer (C2) orchestrating all toolchains."""
    
    def __init__(self, pool: WorkerPool):
        self.pool = pool
        self.is_running = True

    async def custom_search(self, trends: bool = False):
        """Module: Targeted semantic searches and trend parsing."""
        AppState.STOP_REQUESTED = False
        print_header("REAL-TIME TREND DISCOVERY" if trends else "CUSTOM SEARCH MODULE")
        q = ""
        if trends:
            log("Engine: Requesting current topological trends...", "TASK")
            results = await self.pool.accs[0].client.get_trends('trending')
            for i, item in enumerate(results[:15], 1): print(f"  {i}. {item.name}")
            sel = get_input("Select Trend Vector #", "b")
            if sel == 'b' or not str(sel).isdigit(): return
            q = results[int(sel)-1].name
        else:
            q = get_input("Target Query String")
            if not q: return

        s, e = get_input("Since (YYYY-MM-DD)"), get_input("Until (YYYY-MM-DD)")
        mx = get_input("Volume Target (Max Tweets)", 50, int)
        dup = get_input("Fault Tolerance (Max Duplicate Pages)", 2, int)
        f_name = get_input("Output Filename", "search_results")

        known_ids, known_texts = DataEngine.extract_signatures(f_name)
        log(f"Cache Initialized: Registered {len(known_ids)} O(1) signatures.", "FILE")
        
        new, _ = await ScrapingEngine.process(self.pool, 0, ScrapeTask(q, mx, known_ids, known_texts, s, e, dup))
        if new:
            log(f"Merge Request: Committing {len(new)} payloads to long-term storage.", "TASK")
            DataEngine.sync_to_disk(DataEngine.load_full(f_name) + new, f_name)
        else:
            log("Stream Empty: No unique payloads discovered in sequence.", "INFO")

    async def historical_interval(self):
        """Module: Reconstructs high-density datasets using chunked intervals."""
        AppState.STOP_REQUESTED = False
        print_header("HISTORICAL TIMELINE RECONSTRUCTION")
        q = get_input("Target Keyword")
        s_raw = get_input("Start Node (YYYY-MM-DD HH:MM:SS)")
        e_raw = get_input("End Node (YYYY-MM-DD HH:MM:SS)")
        if not all([q, s_raw, e_raw]): return
        
        try:
            s_dt = datetime.strptime(s_raw, "%Y-%m-%d %H:%M:%S")
            e_dt = datetime.strptime(e_raw, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            log("Timestamp format breached. Must match YYYY-MM-DD HH:MM:SS", "ERROR")
            return
            
        hrs, jit = get_input("Time-Slice Width (Hrs)", 2.0, float), get_input("Variance Jitter (Min)", 20, int)
        mx, dup = get_input("Max Extraction/Slice", 25, int), get_input("Fault Tolerance (Max Dup Pages)", 1, int)
        f_name = get_input("Archive Filename", "history_archive")

        ids, texts = DataEngine.extract_signatures(f_name)
        data = DataEngine.load_full(f_name)
        log(f"Profile Loaded: {len(data)} existing nodes for '{q}'.", "FILE")
        
        timeline, cursor = [], s_dt
        while cursor < e_dt:
            delta = timedelta(hours=hrs) + timedelta(minutes=random.randint(-jit, jit))
            end_t = min(cursor + max(timedelta(minutes=10), delta), e_dt)
            timeline.append((cursor, end_t)); cursor = end_t + timedelta(seconds=1)

        log(f"Planner Active: Architecture generated for {len(timeline)} time-slices.", "TASK")
        for i in range(0, len(timeline), len(self.pool.accs)):
            if not self.is_running or AppState.STOP_REQUESTED: break
            tasks = timeline[i : i + len(self.pool.accs)]
            
            async def chunk_runner(its, off):
                t_q = f"{q} since_time:{int(its[0].timestamp())} until_time:{int(its[1].timestamp())}"
                log(f"Executing Node: {its[0].strftime('%Y-%m-%d %H:%M')} ➔ {its[1].strftime('%Y-%m-%d %H:%M')}", "TASK")
                return await ScrapingEngine.process(self.pool, off, ScrapeTask(t_q, mx, ids, texts, None, None, dup))

            results = await asyncio.gather(*(chunk_runner(it, j) for j, it in enumerate(tasks)), return_exceptions=True)
            for res in results:
                if not isinstance(res, Exception) and res[0]: data.extend(res[0])
            # Deduplicate the accumulated list in-place before flushing to disk,
            # guarding against any parallel-task race that slipped a duplicate through.
            deduped_dict = {}
            for row in data:
                rid = str(row.get('tweet_id', ''))
                txt = row.get('text', '').strip()
                if not rid or not txt: continue
                
                if txt not in deduped_dict:
                    deduped_dict[txt] = row
                else:
                    existing_rid = str(deduped_dict[txt].get('tweet_id', ''))
                    if 'E+' in existing_rid.upper() and 'E+' not in rid.upper():
                        deduped_dict[txt] = row
            data = list(deduped_dict.values())
            DataEngine.sync_to_disk(data, f_name)
            
            if i + len(self.pool.accs) < len(timeline): await asyncio.sleep(2)

    async def continuous_poll(self):
        """Module: Persistent surveillance over specific infrastructure keywords."""
        AppState.STOP_REQUESTED = False
        print_header("REAL-TIME PERSISTENT MONITORING")
        q_list = [x.strip() for x in get_input("Infrastructure Tags (comma-sep)").split(',') if x.strip()]
        goal, wait = get_input("Saturation/Tag", 500, int), get_input("Pulse Interval (min)", 20.0, float)
        mx_swp = get_input("Extraction Limit/Pulse", 40, int)
        dup = get_input("Fault Tolerance", 1, int)
        f_name = get_input("Database Stream", "polling_live")

        data = DataEngine.load_full(f_name)
        ids, texts = DataEngine.extract_signatures(f_name)
        stats = {q: sum(1 for r in data if r.get('keyword') == q) for q in q_list}

        try:
            while self.is_running and any(v < goal for v in stats.values()) and not AppState.STOP_REQUESTED:
                worklist = [q for q in q_list if stats[q] < goal]
                for i in range(0, len(worklist), len(self.pool.accs)):
                    if AppState.STOP_REQUESTED: break
                    chunk = worklist[i : i + len(self.pool.accs)]
                    results = await asyncio.gather(*(ScrapingEngine.process(self.pool, j, 
                        ScrapeTask(q, mx_swp, ids, texts, None, None, dup)) for j, q in enumerate(chunk)))
                    
                    for k, res in enumerate(results):
                        stats[chunk[k]] += res[1]; data.extend(res[0])
                    # Deduplicate before flush to prevent parallel races from writing dupes.
                    deduped_dict_p = {}
                    for row in data:
                        rid = str(row.get('tweet_id', ''))
                        txt = row.get('text', '').strip()
                        if not rid or not txt: continue
                        
                        if txt not in deduped_dict_p:
                            deduped_dict_p[txt] = row
                        else:
                            existing_rid = str(deduped_dict_p[txt].get('tweet_id', ''))
                            if 'E+' in existing_rid.upper() and 'E+' not in rid.upper():
                                deduped_dict_p[txt] = row
                    data = list(deduped_dict_p.values())
                    DataEngine.sync_to_disk(data, f_name)
                
                if AppState.STOP_REQUESTED: break
                log(f"Pulse Wave Complete. Hibernate requested for {wait}m...", "WAIT")
                try:
                    for _ in range(int(wait * 60)):
                        if AppState.STOP_REQUESTED: break
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    log("Hibernation interrupted. Returning to main console.", "WARNING")
                    break
        except KeyboardInterrupt: log("Continuous poll manually aborted.", "WAIT")

    async def recalculate_historical(self):
        """Module: Recount tweets_per_date for every row in an existing archive."""
        AppState.STOP_REQUESTED = False
        print_header("RECALCULATE TWEETS PER DATE")

        # ── Pick archive ──────────────────────────────────────────────────────
        root = os.path.dirname(os.path.abspath(__file__))
        csv_files = sorted(glob.glob(os.path.join(root, '*.csv')))
        if csv_files:
            print(f" {Theme.BOLD}DETECTED ARCHIVES:{Theme.RESET}")
            for i, fp in enumerate(csv_files, 1):
                size = os.path.getsize(fp)
                print(f"  {Theme.YELLOW}[{i}]{Theme.RESET} {os.path.basename(fp)}  ({size // 1024} KB)")
            sel = get_input("Select archive # (or type filename without .csv)", "1")
            if str(sel).isdigit() and 1 <= int(sel) <= len(csv_files):
                f_name = os.path.splitext(csv_files[int(sel) - 1])[0]
            else:
                f_name = sel if sel else None
        else:
            f_name = get_input("Archive Filename (no extension)")

        if not f_name:
            log("No archive selected. Aborting.", "ERROR")
            return

        data = DataEngine.load_full(f_name)
        if not data:
            log(f"Archive '{f_name}' is empty or not found.", "ERROR")
            return

        # ── Show BEFORE counts ────────────────────────────────────────────────
        before: Counter = Counter()
        for row in data:
            before[str(row.get('datetime', ''))[:10]] += 1

        print(f"\n  {Theme.BOLD}Current tweets_per_date (before recount):{Theme.RESET}")
        for date, cnt in sorted(before.items()):
            bar = '█' * min(cnt // 2, 50)
            print(f"    {Theme.CYAN}{date}{Theme.RESET}  {bar} {cnt}")

        # ── Deduplicate by exact text and fix Excel corrupted IDs ─────────────
        deduped_dict = {}
        for row in data:
            rid = str(row.get('tweet_id', ''))
            txt = row.get('text', '').strip()
            if not rid or not txt: continue
            
            if txt not in deduped_dict:
                deduped_dict[txt] = row
            else:
                # If we already have it, but the existing one has an Excel corrupted ID
                # (e.g. 2.04E+18) and the new one is valid, replace it!
                existing_rid = str(deduped_dict[txt].get('tweet_id', ''))
                if 'E+' in existing_rid.upper() and 'E+' not in rid.upper():
                    deduped_dict[txt] = row
                    
        deduped_data = list(deduped_dict.values())
        
        dupes_removed = len(data) - len(deduped_data)
        if dupes_removed > 0:
            log(f"Removed {dupes_removed} duplicate records.", "TASK")
            data = deduped_data

        # ── Recount via sync_to_disk (it recalculates tweets_per_date internally)
        log(f"Recounting {len(data)} records across {len(before)} date(s)...", "TASK")
        DataEngine.sync_to_disk(data, f_name)

        # ── Show AFTER counts ─────────────────────────────────────────────────
        after_data = DataEngine.load_full(f_name)
        after: Counter = Counter()
        for row in after_data:
            after[str(row.get('datetime', ''))[:10]] += 1

        print(f"\n  {Theme.BOLD}Updated tweets_per_date (after recount):{Theme.RESET}")
        for date, cnt in sorted(after.items()):
            bar = '█' * min(cnt // 2, 50)
            print(f"    {Theme.CYAN}{date}{Theme.RESET}  {bar} {cnt}")

        log(f"Done. {len(data)} records updated in '{os.path.basename(DataEngine._path(f_name))}'.", "SUCCESS")

    async def add_new_account(self):
        """Module: Expand rotation pool interactively."""
        print_header("SECURE ACCOUNT REGISTRATION")
        root = os.path.dirname(os.path.abspath(__file__))
        idx = len(self.pool.accs) + 1
        while any(os.path.basename(a.cookies_file) == f"twitter_cookies_{idx}.json" for a in self.pool.accs): idx += 1
        name = os.path.join(root, f"twitter_cookies_{idx if idx > 1 else ''}.json")
        if idx == 1: name = os.path.join(root, "twitter_cookies.json")
        
        acc = TwitterAccount(len(self.pool.accs), name)
        if await acc.auth(): self.pool.accs.append(acc); log(f"Node {acc.idx} integrated into fleet.", "AUTH")

    async def remove_account(self):
        """Module: Decommission and wipe an account from the pool."""
        print_header("NODE DECOMMISSION UTILITY")
        if not self.pool.accs:
            log("Fleet is currently empty. No nodes to remove.", "ERROR"); return
            
        for i, acc in enumerate(self.pool.accs, 1):
            status = "Ready" if acc.available() else f"Cooldown ({acc.resume_until.strftime('%H:%M:%S')})"
            print(f"  {i}. Node {acc.idx} (DB: {os.path.basename(acc.cookies_file)}) │ Status: {status}")
            
        sel = get_input("Target Node # for termination (or 'b' to back)", "b")
        if sel == 'b' or not str(sel).isdigit(): return
        
        idx = int(sel) - 1
        if 0 <= idx < len(self.pool.accs):
            acc = self.pool.accs.pop(idx)
            log(f"Node {acc.idx} severed from the cluster.", "SUCCESS")
            
            rem_file = get_input(f"Execute permanent deletion on '{os.path.basename(acc.cookies_file)}'? (y/n)", "n")
            if rem_file.lower() == 'y':
                try:
                    os.remove(acc.cookies_file)
                    log("Session keys permanently purged.", "FILE")
                except Exception as e:
                    log(f"Purge Fault: {e}", "ERROR")
        else:
            log("Invalid Node Designation.", "ERROR")

# =============================================================================
# REGION: ENTRY POINT
# =============================================================================

async def main():
    """Bootstraps the framework, establishes the account fleet, and engages the C2 loop."""
    root = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(root, 'twitter_cookies*.json')))
    accounts = []
    
    for i, f in enumerate(files):
        acc = TwitterAccount(i, f)
        if await acc.auth(): accounts.append(acc)
    
    dash = AnalyticaDashboard(WorkerPool(accounts))
    MENU = {
        "1": ("Trends Discovery",        "Auto-fetch and scrape current trending vectors on X.",                lambda: dash.custom_search(trends=True)),
        "2": ("Custom Search",            "Perform a targeted semantic search with date filters.",              dash.custom_search),
        "3": ("Historical Deep-Dive",     "Reconstruct granular timelines using time-chunking.",                dash.historical_interval),
        "4": ("Recalc Historical",        "Re-plan an archive: extend, gap-fill, or change date/amount.",      dash.recalculate_historical),
        "5": ("Persistent Monitor",       "Sustain real-time architectural polling of keywords.",               dash.continuous_poll),
        "6": ("Link New Account",         "Inject a fresh Twitter identity into the rotation fleet.",           dash.add_new_account),
        "7": ("Decommission Account",     "De-link and securely wipe an account cookie file.",                 dash.remove_account),
        "0": ("Terminate Console",        "Safely flush memory queues and power down all tools.",              None)
    }

    while dash.is_running:
        print(f"\n{Theme.BOLD}{Theme.CYAN}    --- TWITTER ANALYTICA DASHBOARD v2.0 ---{Theme.RESET}")
        print(f"{Theme.BLUE}{'─'*AppConfig.DASHBOARD_WIDTH}{Theme.RESET}")
        
        print(f" {Theme.BOLD}DATA EXTRACTION MATRIX:{Theme.RESET}")
        for k in ["1", "2", "3", "4", "5"]:
            n, d, _ = MENU[k]
            print(f"  {Theme.YELLOW}[{k}]{Theme.RESET} {Theme.BOLD}{n:<22}{Theme.RESET} {Theme.CYAN}➔{Theme.RESET} {d}")
        
        print(f"\n {Theme.BOLD}FLEET MANAGEMENT:{Theme.RESET}")
        for k in ["6", "7"]:
            n, d, _ = MENU[k]
            print(f"  {Theme.YELLOW}[{k}]{Theme.RESET} {Theme.BOLD}{n:<22}{Theme.RESET} {Theme.CYAN}➔{Theme.RESET} {d}")
            
        print(f"\n  {Theme.YELLOW}[0]{Theme.RESET} {Theme.BOLD}Terminate Console{Theme.RESET}")
        print(f"{Theme.BLUE}{'─'*AppConfig.DASHBOARD_WIDTH}{Theme.RESET}")
        
        try:
            ch = get_input("SYSTEM READY. AWAITING DIRECTIVE")
            if ch == '0' or str(ch).lower() == 'q': break
            if ch in MENU:
                try:
                    await MENU[ch][2]()
                except KeyboardInterrupt:
                    log("Operation interrupted. Returning to main menu...", "WAIT")
                finally:
                    AppState.STOP_REQUESTED = False
        except KeyboardInterrupt:
            print()
            log("Returning to main menu... (Ctrl+C again to quit)", "WAIT")
            AppState.STOP_REQUESTED = False
            continue

if __name__ == "__main__":
    try: asyncio.run(main())
    except KeyboardInterrupt: pass
    except Exception as e: log(f"Fatal System Halt: {e}", "FATAL")