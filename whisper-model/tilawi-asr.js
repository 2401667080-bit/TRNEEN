// src/levenshtein.ts
function distance(a, b) {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;
  if (a.length > b.length) [a, b] = [b, a];
  const m = a.length;
  const n = b.length;
  const W = 32;
  const nb = m + W - 1 >> 5;
  const HIGH = 2147483648 >>> 0;
  const Peq = [];
  for (let bi = 0; bi < nb; bi++) {
    const mp = /* @__PURE__ */ new Map();
    const start = bi * W;
    const end = Math.min(start + W, m);
    for (let i = start; i < end; i++) {
      const c = a.charCodeAt(i);
      mp.set(c, ((mp.get(c) ?? 0) | 1 << i - start) >>> 0);
    }
    Peq.push(mp);
  }
  const Pv = new Array(nb).fill(4294967295 >>> 0);
  const Mv = new Array(nb).fill(0);
  const lastBits = m - (nb - 1) * W;
  const lastScoreBit = 1 << lastBits - 1 >>> 0;
  let score = m;
  for (let j = 0; j < n; j++) {
    const c = b.charCodeAt(j);
    let hin = 1;
    for (let bi = 0; bi < nb; bi++) {
      let Eq = Peq[bi].get(c) ?? 0;
      const pv = Pv[bi], mv = Mv[bi];
      const Xv = (Eq | mv) >>> 0;
      if (hin < 0) Eq = (Eq | 1) >>> 0;
      const Xh = ((((Eq & pv) >>> 0) + pv >>> 0 ^ pv) >>> 0 | Eq) >>> 0;
      let Ph = (mv | ~(Xh | pv) >>> 0) >>> 0;
      let Mh = (pv & Xh) >>> 0;
      if (bi === nb - 1) {
        if (Ph & lastScoreBit) score++;
        else if (Mh & lastScoreBit) score--;
      }
      let hout = 0;
      if (Ph & HIGH) hout = 1;
      else if (Mh & HIGH) hout = -1;
      Ph = Ph << 1 >>> 0;
      Mh = Mh << 1 >>> 0;
      if (hin < 0) Mh = (Mh | 1) >>> 0;
      else if (hin > 0) Ph = (Ph | 1) >>> 0;
      Pv[bi] = (Mh | ~(Xv | Ph) >>> 0) >>> 0;
      Mv[bi] = (Ph & Xv) >>> 0;
      hin = hout;
    }
  }
  return score;
}
function ratio(a, b) {
  const lenSum = a.length + b.length;
  if (lenSum === 0) return 1;
  return (lenSum - distance(a, b)) / lenSum;
}
function semiGlobalDistance(query, ref) {
  if (query.length === 0) return 0;
  if (ref.length === 0) return query.length;
  const m = query.length;
  const n = ref.length;
  const W = 32;
  const nb = m + W - 1 >> 5;
  const HIGH = 2147483648 >>> 0;
  const Peq = [];
  for (let bi = 0; bi < nb; bi++) {
    const mp = /* @__PURE__ */ new Map();
    const start = bi * W;
    const end = Math.min(start + W, m);
    for (let i = start; i < end; i++) {
      const c = query.charCodeAt(i);
      mp.set(c, ((mp.get(c) ?? 0) | 1 << i - start) >>> 0);
    }
    Peq.push(mp);
  }
  const Pv = new Array(nb).fill(4294967295 >>> 0);
  const Mv = new Array(nb).fill(0);
  const lastBits = m - (nb - 1) * W;
  const lastScoreBit = 1 << lastBits - 1 >>> 0;
  let score = m;
  let best = m;
  for (let j = 0; j < n; j++) {
    const c = ref.charCodeAt(j);
    let hin = 0;
    for (let bi = 0; bi < nb; bi++) {
      let Eq = Peq[bi].get(c) ?? 0;
      const pv = Pv[bi], mv = Mv[bi];
      const Xv = (Eq | mv) >>> 0;
      if (hin < 0) Eq = (Eq | 1) >>> 0;
      const Xh = ((((Eq & pv) >>> 0) + pv >>> 0 ^ pv) >>> 0 | Eq) >>> 0;
      let Ph = (mv | ~(Xh | pv) >>> 0) >>> 0;
      let Mh = (pv & Xh) >>> 0;
      if (bi === nb - 1) {
        if (Ph & lastScoreBit) score++;
        else if (Mh & lastScoreBit) score--;
      }
      let hout = 0;
      if (Ph & HIGH) hout = 1;
      else if (Mh & HIGH) hout = -1;
      Ph = Ph << 1 >>> 0;
      Mh = Mh << 1 >>> 0;
      if (hin < 0) Mh = (Mh | 1) >>> 0;
      else if (hin > 0) Ph = (Ph | 1) >>> 0;
      Pv[bi] = (Mh | ~(Xv | Ph) >>> 0) >>> 0;
      Mv[bi] = (Ph & Xv) >>> 0;
      hin = hout;
    }
    if (score < best) best = score;
  }
  return best;
}
function fragmentScore(query, ref) {
  if (query.length === 0) return 1;
  return Math.max(0, 1 - semiGlobalDistance(query, ref) / query.length);
}

// src/quran-db.ts
var _BSM_PHONEMES_JOINED = "bismi allahi arraHmaani arraHiimi";
var _BSM_PHONEME_TOKENS = "b i s m i | a l l a h i | a r r a H m aa n i | a r r a H ii m i".split(" ");
var _BSM_ARABIC_WORDS = ["\u0628\u0633\u0645", "\u0627\u0644\u0644\u0647", "\u0627\u0644\u0631\u062D\u0645\u0646", "\u0627\u0644\u0631\u062D\u064A\u0645"];
var JOINT_TOP_K_LEVENSHTEIN = 18;
var JOINT_TOP_SURAHS = 32;
var JOINT_MAX_SPAN = 6;
var JOINT_FRAGMENT_BLEND = 0.82;
var JOINT_PREFIX_MAX_SPAN = 7;
var JOINT_PREFIX_MIN_CHARS = 34;
var JOINT_PREFIX_MIN_SCORE = 0.5;
var JOINT_PREFIX_MARGIN = -0.02;
var JOINT_GLOBAL_SPAN_MIN_CHARS = 80;
var JOINT_GLOBAL_SPAN_MIN_SCORE = 0.54;
var JOINT_GLOBAL_SPAN_MARGIN = -0.015;
var JOINT_GLOBAL_SPAN_SHORTLIST = 320;
var JOINT_OPENING_COLLAPSE_MIN_CHARS = 34;
var JOINT_OPENING_COLLAPSE_MAX_CHARS = 115;
var JOINT_OPENING_COLLAPSE_MIN_SCORE = 0.5;
var JOINT_SHORT_OPENING_MAX_CHARS = 10;
var JOINT_SHORT_OPENING_MIN_QUERY_CHARS = 8;
var JOINT_SHORT_OPENING_MIN_SCORE = 0.48;
var JOINT_SHORT_OPENING_MARGIN = -0.18;
function partialRatio(short, long) {
  if (!short || !long)
    return 0;
  if (short.length > long.length)
    [short, long] = [long, short];
  const window = short.length;
  let best = 0;
  for (let i = 0; i <= Math.max(0, long.length - window); i++) {
    const r = ratio(short, long.slice(i, i + window));
    if (r > best) {
      best = r;
      if (best === 1)
        break;
    }
  }
  return best;
}
var QuranDB = class _QuranDB {
  constructor(data, tokenEncoder, ctcTokenTable) {
    this._byRef = /* @__PURE__ */ new Map();
    this._bySurah = /* @__PURE__ */ new Map();
    this._jointPrefixSpans = null;
    this._jointGlobalSpans = null;
    this.tokenEncoder = tokenEncoder;
    this.ctcTokenTable = ctcTokenTable;
    this.verses = data;
    for (const v of data) {
      this._byRef.set(`${v.surah}:${v.ayah}`, v);
      const arr = this._bySurah.get(v.surah) ?? [];
      arr.push(v);
      this._bySurah.set(v.surah, arr);
      v.phoneme_tokens = v.phoneme_tokens?.length ? v.phoneme_tokens : v.phonemes.trim().split(/\s+/).filter(Boolean);
      if (v.ayah === 1 && v.surah !== 1 && v.surah !== 9 && v.phonemes_joined.startsWith(_BSM_PHONEMES_JOINED)) {
        const stripped = v.phonemes_joined.slice(_BSM_PHONEMES_JOINED.length).trim();
        v.phonemes_joined_no_bsm = stripped || null;
        let strippedTokens = v.phoneme_tokens.slice(_BSM_PHONEME_TOKENS.length);
        if (strippedTokens[0] === "|") {
          strippedTokens = strippedTokens.slice(1);
        }
        v.phoneme_tokens_no_bsm = strippedTokens.length ? strippedTokens : null;
      } else {
        v.phonemes_joined_no_bsm = null;
        v.phoneme_tokens_no_bsm = null;
      }
      if (v.ayah === 1 && v.surah !== 1 && v.surah !== 9 && this._startsWithArabicBismillah(v.phoneme_words)) {
        const strippedWords = v.phoneme_words.slice(_BSM_ARABIC_WORDS.length);
        v.phonemes_joined_no_bsm = strippedWords.length ? strippedWords.join(" ") : null;
        const bsmTokenEnd = v.word_token_ends?.[_BSM_ARABIC_WORDS.length - 1] ?? 0;
        v.phoneme_tokens_no_bsm = bsmTokenEnd > 0 && v.phoneme_tokens.length > bsmTokenEnd ? v.phoneme_tokens.slice(bsmTokenEnd) : null;
        v.phoneme_token_ids_no_bsm = bsmTokenEnd > 0 && (v.phoneme_token_ids?.length ?? 0) > bsmTokenEnd ? v.phoneme_token_ids.slice(bsmTokenEnd) : null;
      }
      v.phonemes_joined_ns = v.phonemes_joined.replace(/ /g, "");
      v.phonemes_joined_no_bsm_ns = v.phonemes_joined_no_bsm ? v.phonemes_joined_no_bsm.replace(/ /g, "") : null;
      if (this.tokenEncoder) {
        v.phoneme_token_ids = this.tokenEncoder.encodeRawPhonemes(v.phonemes);
        v.phoneme_token_ids_no_bsm = v.phoneme_tokens_no_bsm ? this.tokenEncoder.encodeRawPhonemes(v.phoneme_tokens_no_bsm.join(" ")) : null;
      } else {
        v.phoneme_token_ids = v.phoneme_token_ids ?? [];
        v.phoneme_token_ids_no_bsm = v.phoneme_token_ids_no_bsm ?? null;
      }
      v.word_token_ends = v.word_token_ends?.length ? v.word_token_ends : this._computeWordTokenEnds(v.phoneme_tokens);
    }
  }
  get totalVerses() {
    return this.verses.length;
  }
  get surahCount() {
    return this._bySurah.size;
  }
  getVerse(surah, ayah) {
    return this._byRef.get(`${surah}:${ayah}`);
  }
  getSurah(surah) {
    return this._bySurah.get(surah) ?? [];
  }
  getNextVerse(surah, ayah) {
    const verses = this._bySurah.get(surah) ?? [];
    for (let i = 0; i < verses.length; i++) {
      if (verses[i].ayah === ayah) {
        if (i + 1 < verses.length)
          return verses[i + 1];
        const nextSurah = this._bySurah.get(surah + 1) ?? [];
        return nextSurah[0];
      }
    }
    return void 0;
  }
  _startsWithArabicBismillah(words) {
    if (!words || words.length <= _BSM_ARABIC_WORDS.length)
      return false;
    return _BSM_ARABIC_WORDS.every((word, index) => words[index] === word);
  }
  /** Return candidates for verses whose non-Bsm phoneme token IDs are short (≤ maxTokens). */
  getShortVerseCandidates(maxTokens = 15) {
    const result = [];
    for (const v of this.verses) {
      const ids = v.phoneme_token_ids_no_bsm ?? v.phoneme_token_ids ?? [];
      if (ids.length === 0 || ids.length > maxTokens)
        continue;
      result.push({
        surah: v.surah,
        ayah: v.ayah,
        text: v.phonemes_joined,
        phonemes_joined: v.phonemes_joined,
        phoneme_token_ids: ids,
        stage_a_score: 0,
        raw_score: 0,
        bonus: 0,
        kind: "single"
      });
    }
    return result;
  }
  search(text, topK = 5) {
    const scored = [];
    for (const v of this.verses) {
      const score = ratio(text, v.phonemes_joined);
      scored.push({ ...v, score });
    }
    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK);
  }
  retrieveCandidates(text, { maxSpan = 4, hint = null, singleLimit = 32, topSurahs = 3, spanLimit = 32 } = {}) {
    if (!text.trim()) {
      return { singles: [], spans: [], combined: [] };
    }
    const bonuses = this._continuationBonuses(hint);
    const textWords = text.split(/\s+/).filter(Boolean);
    const noSpaceText = text.replace(/ /g, "");
    const scored = [];
    for (const v of this.verses) {
      let raw = ratio(text, v.phonemes_joined);
      const verseWords = v.phoneme_words;
      const sharedWordCount = Math.min(textWords.length, verseWords.length);
      if (sharedWordCount > 0) {
        const textPrefix = textWords.slice(0, sharedWordCount).join(" ");
        const versePrefix = verseWords.slice(0, sharedWordCount).join(" ");
        raw = Math.max(raw, ratio(textPrefix, versePrefix));
      }
      if (noSpaceText.length <= 10) {
        raw = Math.max(raw, this._shortQueryBoost(noSpaceText, v));
      }
      if (v.phonemes_joined_no_bsm) {
        raw = Math.max(raw, ratio(text, v.phonemes_joined_no_bsm));
        if (noSpaceText.length <= 10) {
          raw = Math.max(raw, this._shortQueryBoost(noSpaceText, v, true));
        }
      }
      const bonus = bonuses.get(`${v.surah}:${v.ayah}`) ?? 0;
      if (bonus > 0) {
        const sp = _QuranDB._suffixPrefixScore(text, v.phonemes_joined);
        raw = Math.max(raw, sp);
      }
      scored.push([v, raw, bonus, Math.min(raw + bonus, 1)]);
    }
    scored.sort((a, b) => b[3] - a[3]);
    const pass2Surahs = [];
    for (let i = 0; i < scored.length && pass2Surahs.length < topSurahs; i++) {
      const surah = scored[i][0].surah;
      if (!pass2Surahs.includes(surah)) {
        pass2Surahs.push(surah);
      }
    }
    if (noSpaceText.length >= 8) {
      let resorted = false;
      for (let i = 0; i < scored.length; i++) {
        const [v, raw, bonus] = scored[i];
        if (noSpaceText.length >= (v.phonemes_joined_ns?.length ?? 0) * 0.8)
          continue;
        let frag = fragmentScore(noSpaceText, v.phonemes_joined_ns ?? "");
        if (v.phonemes_joined_no_bsm_ns) {
          frag = Math.max(frag, fragmentScore(noSpaceText, v.phonemes_joined_no_bsm_ns));
        }
        if (frag > raw) {
          const boosted = raw + (frag - raw) * 0.7;
          scored[i] = [v, boosted, bonus, Math.min(boosted + bonus, 1)];
          resorted = true;
        }
      }
      if (resorted)
        scored.sort((a, b) => b[3] - a[3]);
    }
    const singles = scored.slice(0, singleLimit).map(([v, raw, bonus, total]) => this._candidateFromVerse(v, raw, bonus, total));
    const spans = [];
    for (let surahRank = 0; surahRank < pass2Surahs.length; surahRank++) {
      const surah = pass2Surahs[surahRank];
      const verses = this._bySurah.get(surah) ?? [];
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= maxSpan; span++) {
          if (i + span > verses.length)
            break;
          const chunk = verses.slice(i, i + span);
          const spanText = this._joinedSpanPhonemes(chunk);
          let raw = ratio(text, spanText);
          const spanWords = spanText.split(/\s+/).filter(Boolean);
          const sharedWordCount = Math.min(textWords.length, spanWords.length);
          if (sharedWordCount > 0) {
            const textPrefix = textWords.slice(0, sharedWordCount).join(" ");
            const spanPrefix = spanWords.slice(0, sharedWordCount).join(" ");
            raw = Math.max(raw, ratio(textPrefix, spanPrefix));
          }
          const bonus = bonuses.get(`${chunk[0].surah}:${chunk[0].ayah}`) ?? 0;
          const score = Math.min(raw + bonus, 1);
          spans.push(this._candidateFromSpan(chunk, raw, bonus, score, surahRank));
        }
      }
    }
    spans.sort((a, b) => b.stage_a_score - a.stage_a_score);
    return {
      singles,
      spans: spans.slice(0, spanLimit),
      combined: singles.concat(spans.slice(0, spanLimit))
    };
  }
  matchVerse(text, threshold = 0.3, maxSpan = 3, hint = null, returnTopK = 0) {
    const retrieved = this.retrieveCandidates(text, {
      maxSpan,
      hint,
      singleLimit: Math.max(returnTopK, 5),
      topSurahs: 20,
      spanLimit: 64
    });
    const ranked = retrieved.combined.slice().sort((a, b) => b.stage_a_score - a.stage_a_score);
    const best = ranked[0];
    if (!best || best.stage_a_score < threshold) {
      return null;
    }
    const result = {
      surah: best.surah,
      ayah: best.ayah,
      ayah_end: best.ayah_end,
      text: best.text,
      phonemes_joined: best.phonemes_joined,
      score: best.stage_a_score,
      raw_score: best.raw_score,
      bonus: best.bonus
    };
    if (returnTopK > 0) {
      result.runners_up = retrieved.singles.slice(0, returnTopK).map((candidate) => ({
        surah: candidate.surah,
        ayah: candidate.ayah,
        raw_score: Math.round(candidate.raw_score * 1e3) / 1e3,
        bonus: Math.round(candidate.bonus * 1e3) / 1e3,
        score: Math.round(candidate.stage_a_score * 1e3) / 1e3,
        phonemes_joined: candidate.phonemes_joined.slice(0, 60)
      }));
    }
    return result;
  }
  matchPhonemeTextJoint03(text, topK = JOINT_TOP_K_LEVENSHTEIN) {
    return this._joint02MatchPhonemeText(text, topK);
  }
  bestJoint03Match(text) {
    const top = this._joint02MatchPhonemeText(text, JOINT_TOP_K_LEVENSHTEIN);
    if (!top.length)
      return null;
    const best = top[0];
    const bestScore = best.score;
    let winner;
    const shortOpening = this._jointShortOpeningSpanCandidate(text, bestScore);
    if (shortOpening) {
      winner = shortOpening;
    } else {
      const bestIsLateSpan = best.ayah_end != null && best.ayah > 1;
      const lowConfidence = bestScore < 0.62;
      if (!bestIsLateSpan && !lowConfidence) {
        winner = best;
      } else {
        const noSpaceLen = text.replace(/ /g, "").length;
        const prefix = this._jointSurahPrefixCandidates(text);
        const globalSpan = this._jointGlobalSpanCandidates(text);
        const candidates = [best].concat(prefix.filter((p) => p.score >= bestScore + JOINT_PREFIX_MARGIN)).concat(globalSpan.filter((g) => g.score >= bestScore + JOINT_GLOBAL_SPAN_MARGIN));
        candidates.sort((a, b) => b.score - a.score);
        const chosen = candidates[0];
        if (noSpaceLen >= JOINT_OPENING_COLLAPSE_MIN_CHARS && noSpaceLen <= JOINT_OPENING_COLLAPSE_MAX_CHARS && best.ayah_end != null && best.ayah > 1) {
          const sameSurahPrefix = prefix.filter((p) => p.surah === best.surah && p.score >= JOINT_OPENING_COLLAPSE_MIN_SCORE && (p.ayah_end == null || best.ayah_end == null || p.ayah_end >= best.ayah_end)).sort((a, b) => b.score - a.score);
          if (sameSurahPrefix.length > 0)
            winner = sameSurahPrefix[0];
        }
        if (!winner)
          winner = chosen;
      }
    }
    this._annotateConfidence(winner, top, text);
    return winner;
  }
  /**
   * Attach confidence signals to a chosen match without affecting selection:
   *  - `margin`: score() gap to the best DISTINCT verse in the ranked list.
   *    A large margin means the match is unambiguous even when its absolute
   *    score is depressed (e.g. a short fragment of a long verse).
   *  - `frag_fit`: pure semi-global fragment fit of the query against the
   *    match's phonemes (1.0 = query is an exact substring of the match).
   * These let the trust gate accept distinctive fragments that the flat
   * absolute-score threshold alone would wrongly reject.
   */
  _annotateConfidence(winner, top, text) {
    const noSpaceText = text.replace(/ /g, "");
    const refNs = (winner.phonemes_joined ?? "").replace(/ /g, "");
    const winnerFit = refNs && noSpaceText ? fragmentScore(noSpaceText, refNs) : 0;
    winner.frag_fit = _QuranDB._round4(winnerFit);
    let bestOtherScore = 0;
    let bestOtherFit = 0;
    for (const c of top) {
      if (this._sameOrOverlapping(c, winner))
        continue;
      if (c.score > bestOtherScore)
        bestOtherScore = c.score;
      const cns = (c.phonemes_joined ?? "").replace(/ /g, "");
      if (cns && noSpaceText) {
        const f = fragmentScore(noSpaceText, cns);
        if (f > bestOtherFit)
          bestOtherFit = f;
      }
    }
    winner.margin = _QuranDB._round4(winner.score - bestOtherScore);
    winner.fit_margin = _QuranDB._round4(winnerFit - bestOtherFit);
  }
  /** True when candidate c is the winner or a same-surah span that overlaps it. */
  _sameOrOverlapping(c, w) {
    if (c.surah !== w.surah)
      return false;
    const cEnd = c.ayah_end ?? c.ayah;
    const wEnd = w.ayah_end ?? w.ayah;
    return c.ayah <= wEnd && cEnd >= w.ayah;
  }
  bestJoint03MatchForHypotheses(hypotheses) {
    let best = null;
    for (const transcript of hypotheses) {
      const match = this.bestJoint03Match(transcript);
      if (!match)
        continue;
      if (!best || match.score > best.match.score) {
        best = { match, transcript };
      }
    }
    return best;
  }
  _joint02MatchPhonemeText(phonemeText, topK = JOINT_TOP_K_LEVENSHTEIN) {
    if (!phonemeText.trim())
      return [];
    const noSpaceText = phonemeText.replace(/ /g, "");
    const scored = [];
    for (const verse of this._jointCandidateVerses(noSpaceText)) {
      const ref = verse.phonemes_joined;
      if (!ref)
        continue;
      let raw = ratio(phonemeText, ref);
      if (noSpaceText.length <= 10) {
        raw = Math.max(raw, this._shortQueryBoost(noSpaceText, verse));
      }
      const noBsm = verse.phonemes_joined_no_bsm;
      if (noBsm) {
        raw = Math.max(raw, ratio(phonemeText, noBsm));
        if (noSpaceText.length <= 10) {
          raw = Math.max(raw, this._shortQueryBoost(noSpaceText, verse, true));
        }
      }
      scored.push([verse, raw, raw]);
    }
    scored.sort((a, b) => b[2] - a[2]);
    const pass2Surahs = [];
    for (const [verse] of scored) {
      if (!pass2Surahs.includes(verse.surah)) {
        pass2Surahs.push(verse.surah);
      }
      if (pass2Surahs.length >= JOINT_TOP_SURAHS)
        break;
    }
    if (noSpaceText.length >= 8) {
      let resorted = false;
      for (let i = 0; i < scored.length; i++) {
        const [verse, raw] = scored[i];
        const refNs = verse.phonemes_joined_ns ?? "";
        if (!refNs || noSpaceText.length >= refNs.length * 0.8)
          continue;
        let frag = fragmentScore(noSpaceText, refNs);
        const noBsmNs = verse.phonemes_joined_no_bsm_ns;
        if (noBsmNs) {
          frag = Math.max(frag, fragmentScore(noSpaceText, noBsmNs));
        }
        if (frag > raw) {
          const boosted = raw + (frag - raw) * JOINT_FRAGMENT_BLEND;
          scored[i] = [verse, boosted, boosted];
          resorted = true;
        }
      }
      if (resorted)
        scored.sort((a, b) => b[2] - a[2]);
    }
    const spanResults = [];
    for (const surahNum of pass2Surahs) {
      const verses = this._bySurah.get(surahNum) ?? [];
      for (let i = 0; i < verses.length; i++) {
        for (let span = 2; span <= JOINT_MAX_SPAN; span++) {
          if (i + span > verses.length)
            break;
          const chunk = verses.slice(i, i + span);
          const spanPhonemes = this._joinedSpanPhonemes(chunk);
          const score = _QuranDB._round4(ratio(phonemeText, spanPhonemes));
          spanResults.push({
            surah: surahNum,
            ayah: chunk[0].ayah,
            ayah_end: chunk[chunk.length - 1].ayah,
            text: chunk.map((verse) => verse.text_uthmani).join(" "),
            phonemes_joined: spanPhonemes,
            score,
            raw_score: score,
            bonus: 0
          });
        }
      }
    }
    const singles = scored.slice(0, Math.max(topK, 32)).map(([verse, raw, boosted]) => {
      const score = _QuranDB._round4(boosted);
      return {
        surah: verse.surah,
        ayah: verse.ayah,
        ayah_end: null,
        text: verse.text_uthmani,
        phonemes_joined: verse.phonemes_joined,
        score,
        raw_score: _QuranDB._round4(raw),
        bonus: 0
      };
    });
    return singles.concat(spanResults).sort((a, b) => b.score - a.score).slice(0, topK);
  }
  _jointSurahPrefixCandidates(phonemeText) {
    if (!phonemeText.trim())
      return [];
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_PREFIX_MIN_CHARS)
      return [];
    const out = [];
    for (const row of this._jointPrefixSpanTable()) {
      const raw = ratio(phonemeText, row.phonemes_joined);
      const frag = fragmentScore(noSpaceText, row.phonemes_joined.replace(/ /g, ""));
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (score < JOINT_PREFIX_MIN_SCORE)
        continue;
      out.push({
        ...row,
        score: _QuranDB._round4(score),
        raw_score: _QuranDB._round4(raw),
        bonus: 0,
        _prefix_rescue: true
      });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, 12);
  }
  _jointGlobalSpanCandidates(phonemeText) {
    if (!phonemeText.trim())
      return [];
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_GLOBAL_SPAN_MIN_CHARS)
      return [];
    const qb = _QuranDB._jointNgrams(noSpaceText, 2);
    const qt = _QuranDB._jointNgrams(noSpaceText, 3);
    const rough = [];
    for (const row of this._jointGlobalSpanTable()) {
      const ov = _QuranDB._intersectionSize(qb, row.bigrams) + 0.48 * _QuranDB._intersectionSize(qt, row.trigrams);
      if (ov > 0)
        rough.push([ov, row]);
    }
    rough.sort((a, b) => b[0] - a[0]);
    const out = [];
    for (const [, row] of rough.slice(0, JOINT_GLOBAL_SPAN_SHORTLIST)) {
      const raw = ratio(phonemeText, row.phonemes);
      const frag = fragmentScore(noSpaceText, row.phonemesNs);
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (score < JOINT_GLOBAL_SPAN_MIN_SCORE)
        continue;
      out.push({
        surah: row.surah,
        ayah: row.ayah,
        ayah_end: row.ayah_end,
        text: this._spanText(row.surah, row.ayah, row.ayah_end),
        phonemes_joined: row.phonemes,
        score: _QuranDB._round4(score),
        raw_score: _QuranDB._round4(raw),
        bonus: 0,
        _global_span_rescue: true
      });
    }
    return out.sort((a, b) => b.score - a.score).slice(0, 12);
  }
  _jointShortOpeningSpanCandidate(phonemeText, bestScore) {
    const noSpaceText = phonemeText.replace(/ /g, "");
    if (noSpaceText.length < JOINT_SHORT_OPENING_MIN_QUERY_CHARS)
      return null;
    let best = null;
    for (const row of this._jointPrefixSpanTable()) {
      const first = this.getVerse(row.surah, 1);
      const second = this.getVerse(row.surah, 2);
      if (!first || !second)
        continue;
      const firstNs = first.phonemes_joined_no_bsm_ns ?? first.phonemes_joined_ns ?? first.phonemes_joined.replace(/ /g, "");
      if (!firstNs || firstNs.length > JOINT_SHORT_OPENING_MAX_CHARS)
        continue;
      if (!noSpaceText.startsWith(firstNs))
        continue;
      const remainder = noSpaceText.slice(firstNs.length);
      const secondNs = second.phonemes_joined_ns ?? second.phonemes_joined.replace(/ /g, "");
      const compareLen = Math.min(secondNs.length, remainder.length, 12);
      if (compareLen >= 4 && ratio(remainder.slice(0, compareLen), secondNs.slice(0, compareLen)) < 0.72) {
        continue;
      }
      const raw = ratio(phonemeText, row.phonemes_joined);
      const frag = fragmentScore(noSpaceText, row.phonemes_joined.replace(/ /g, ""));
      const score = Math.max(raw, raw + (frag - raw) * JOINT_FRAGMENT_BLEND);
      if (score >= JOINT_SHORT_OPENING_MIN_SCORE && score >= bestScore + JOINT_SHORT_OPENING_MARGIN && (!best || score > best.score)) {
        best = {
          ...row,
          score: _QuranDB._round4(score),
          raw_score: _QuranDB._round4(raw),
          bonus: 0,
          _prefix_rescue: true
        };
      }
    }
    return best;
  }
  _jointCandidateVerses(noSpaceText, maxCandidates = 950) {
    if (noSpaceText.length < 4)
      return this.verses;
    const qb = _QuranDB._jointNgrams(noSpaceText, 2);
    const qt = _QuranDB._jointNgrams(noSpaceText, 3);
    if (qb.size === 0 && qt.size === 0)
      return this.verses;
    const scored = [];
    for (let i = 0; i < this.verses.length; i++) {
      const refNs = this.verses[i].phonemes_joined_ns ?? "";
      if (refNs.length < 2)
        continue;
      const ov = _QuranDB._intersectionSize(qb, _QuranDB._jointNgrams(refNs, 2)) + 0.48 * _QuranDB._intersectionSize(qt, _QuranDB._jointNgrams(refNs, 3));
      if (ov > 0)
        scored.push([ov, i]);
    }
    if (scored.length < 80)
      return this.verses;
    scored.sort((a, b) => b[0] - a[0]);
    return scored.slice(0, maxCandidates).map(([, index]) => this.verses[index]);
  }
  _jointPrefixSpanTable() {
    if (this._jointPrefixSpans)
      return this._jointPrefixSpans;
    const spans = [];
    for (const [surahNum, verses] of this._bySurah.entries()) {
      if (!verses.length || verses[0].ayah !== 1)
        continue;
      const maxSpan = Math.min(JOINT_PREFIX_MAX_SPAN, verses.length);
      for (let span = 2; span <= maxSpan; span++) {
        const chunk = verses.slice(0, span);
        spans.push({
          surah: surahNum,
          ayah: 1,
          ayah_end: chunk[chunk.length - 1].ayah,
          text: chunk.map((verse) => verse.text_uthmani).join(" "),
          phonemes_joined: this._joinedSpanPhonemes(chunk),
          score: 0,
          raw_score: 0,
          bonus: 0
        });
      }
    }
    this._jointPrefixSpans = spans;
    return spans;
  }
  _jointGlobalSpanTable() {
    if (this._jointGlobalSpans)
      return this._jointGlobalSpans;
    const spans = [];
    for (const [surahNum, verses] of this._bySurah.entries()) {
      for (let i = 0; i < verses.length; i++) {
        const maxSpan = Math.min(JOINT_PREFIX_MAX_SPAN, verses.length - i);
        for (let span = 2; span <= maxSpan; span++) {
          const chunk = verses.slice(i, i + span);
          const phonemes = this._joinedSpanPhonemes(chunk);
          const phonemesNs = phonemes.replace(/ /g, "");
          spans.push({
            surah: surahNum,
            ayah: chunk[0].ayah,
            ayah_end: chunk[chunk.length - 1].ayah,
            phonemes,
            phonemesNs,
            bigrams: _QuranDB._jointNgrams(phonemesNs, 2),
            trigrams: _QuranDB._jointNgrams(phonemesNs, 3)
          });
        }
      }
    }
    this._jointGlobalSpans = spans;
    return spans;
  }
  _spanText(surah, ayah, ayahEnd) {
    const verses = this._bySurah.get(surah) ?? [];
    return verses.filter((verse) => verse.ayah >= ayah && verse.ayah <= ayahEnd).map((verse) => verse.text_uthmani).join(" ");
  }
  _computeWordTokenEnds(tokens) {
    const ends = [];
    let rawTokenIndex = 0;
    let lastWasBoundary = true;
    for (const token of tokens) {
      rawTokenIndex++;
      if (token === "|") {
        lastWasBoundary = true;
        continue;
      }
      if (lastWasBoundary) {
        ends.push(rawTokenIndex);
      } else {
        ends[ends.length - 1] = rawTokenIndex;
      }
      lastWasBoundary = false;
    }
    return ends;
  }
  _candidateFromVerse(verse, raw, bonus, total) {
    return {
      surah: verse.surah,
      ayah: verse.ayah,
      ayah_end: verse.ayah,
      text: verse.text_uthmani,
      phonemes_joined: verse.phonemes_joined,
      phoneme_token_ids: verse.phoneme_token_ids_no_bsm ?? verse.phoneme_token_ids ?? [],
      stage_a_score: total,
      raw_score: raw,
      bonus,
      kind: "single"
    };
  }
  _candidateFromSpan(chunk, raw, bonus, total, surahRank) {
    const first = chunk[0];
    const spanKey = `${first.surah}:${first.ayah}:${chunk[chunk.length - 1].ayah}`;
    const tokenIds = this.ctcTokenTable?.[spanKey] ?? this._concatenatedSpanTokenIds(chunk);
    return {
      surah: first.surah,
      ayah: first.ayah,
      ayah_end: chunk[chunk.length - 1].ayah,
      text: chunk.map((verse) => verse.text_uthmani).join(" "),
      phonemes_joined: this._joinedSpanPhonemes(chunk),
      phoneme_token_ids: tokenIds,
      stage_a_score: total,
      raw_score: raw,
      bonus,
      kind: "span",
      surah_rank: surahRank
    };
  }
  _joinedSpanPhonemes(chunk) {
    const firstText = chunk[0].phonemes_joined_no_bsm ?? chunk[0].phonemes_joined;
    return [firstText].concat(chunk.slice(1).map((verse) => verse.phonemes_joined)).join(" ");
  }
  _concatenatedSpanTokenIds(chunk) {
    const tokenIds = [];
    const firstIds = chunk[0].phoneme_token_ids_no_bsm ?? chunk[0].phoneme_token_ids ?? [];
    tokenIds.push(...firstIds);
    for (let i = 1; i < chunk.length; i++) {
      tokenIds.push(...chunk[i].phoneme_token_ids ?? []);
    }
    return tokenIds;
  }
  _shortQueryBoost(noSpaceText, verse, useNoBsm = false) {
    const candidate = useNoBsm ? verse.phonemes_joined_no_bsm_ns ?? verse.phonemes_joined_ns ?? "" : verse.phonemes_joined_ns ?? "";
    if (!candidate)
      return 0;
    const prefixWindow = Math.min(candidate.length, noSpaceText.length + 6);
    const prefix = ratio(noSpaceText, candidate.slice(0, prefixWindow));
    const firstWord = useNoBsm ? (verse.phonemes_joined_no_bsm ?? "").split(" ")[0] ?? "" : verse.phoneme_words[0] ?? "";
    const firstWordScore = firstWord ? ratio(noSpaceText, firstWord) : 0;
    return Math.max(prefix, firstWordScore);
  }
  _continuationBonuses(hint) {
    const bonuses = /* @__PURE__ */ new Map();
    if (!hint)
      return bonuses;
    const [hSurah, hAyah] = hint;
    const nv = this._byRef.get(`${hSurah}:${hAyah + 1}`);
    if (nv) {
      bonuses.set(`${hSurah}:${hAyah + 1}`, 0.22);
      if (this._byRef.has(`${hSurah}:${hAyah + 2}`))
        bonuses.set(`${hSurah}:${hAyah + 2}`, 0.12);
      if (this._byRef.has(`${hSurah}:${hAyah + 3}`))
        bonuses.set(`${hSurah}:${hAyah + 3}`, 0.06);
    } else {
      const nextVerses = this._bySurah.get(hSurah + 1) ?? [];
      const bonusValues = [0.22, 0.12, 0.06];
      for (let i = 0; i < Math.min(nextVerses.length, 3); i++) {
        bonuses.set(`${nextVerses[i].surah}:${nextVerses[i].ayah}`, bonusValues[i]);
      }
    }
    return bonuses;
  }
  static _suffixPrefixScore(text, verseText) {
    const wordsT = text.split(" ");
    const wordsV = verseText.split(" ");
    if (wordsT.length < 2 || wordsV.length < 2)
      return 0;
    let best = 0;
    const maxTrim = Math.min(Math.floor(wordsT.length / 2), 4);
    for (let trim = 1; trim <= maxTrim; trim++) {
      const suffix = wordsT.slice(trim).join(" ");
      const n = wordsT.length - trim;
      const prefix = wordsV.slice(0, Math.min(n, wordsV.length)).join(" ");
      best = Math.max(best, ratio(suffix, prefix));
    }
    return best;
  }
  static _jointNgrams(s, n) {
    const out = /* @__PURE__ */ new Set();
    if (s.length < n)
      return out;
    for (let i = 0; i <= s.length - n; i++) {
      out.add(s.slice(i, i + n));
    }
    return out;
  }
  static _intersectionSize(a, b) {
    let count = 0;
    for (const item of a) {
      if (b.has(item))
        count++;
    }
    return count;
  }
  static _round4(value) {
    return Math.round(value * 1e4) / 1e4;
  }
};

// src/normalizer.ts
var DIACRITICS_RE = /[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED\u0640]/g;
var NORM_MAP = {
  "\u0623": "\u0627",
  // أ -> ا
  "\u0625": "\u0627",
  // إ -> ا
  "\u0622": "\u0627",
  // آ -> ا
  "\u0671": "\u0627",
  // ٱ -> ا
  "\u0629": "\u0647",
  // ة -> ه
  "\u0649": "\u064A"
  // ى -> ي
};
function normalizeArabic(text) {
  text = text.replace(/\uFEFF/g, "");
  text = text.replace(DIACRITICS_RE, "");
  text = text.replace(/./g, (ch) => NORM_MAP[ch] ?? ch);
  text = text.split(/\s+/).filter(Boolean).join(" ");
  return text;
}

// src/text-ctc-decode.ts
var WORD_PREFIX = "\u2581";
var TextCTCDecoder = class {
  constructor(vocabJson, blankId) {
    this.vocab = /* @__PURE__ */ new Map();
    let maxId = 0;
    for (const [id, token] of Object.entries(vocabJson)) {
      const numId = Number(id);
      this.vocab.set(numId, token);
      maxId = Math.max(maxId, numId);
    }
    this.blankId = blankId ?? maxId;
  }
  decode(logprobs, timeSteps, vocabSize) {
    const frameIds = [];
    for (let t = 0; t < timeSteps; t++) {
      const offset = t * vocabSize;
      let maxIdx = 0;
      let maxVal = logprobs[offset];
      for (let v = 1; v < vocabSize; v++) {
        const value = logprobs[offset + v];
        if (value > maxVal) {
          maxVal = value;
          maxIdx = v;
        }
      }
      frameIds.push(maxIdx);
    }
    const tokenIds = [];
    let previous = -1;
    for (const id of frameIds) {
      if (id !== previous && id !== this.blankId) {
        tokenIds.push(id);
      }
      previous = id;
    }
    const text = this.tokenIdsToText(tokenIds);
    return {
      text,
      rawPhonemes: text,
      tokenIds
    };
  }
  getBlankId() {
    return this.blankId;
  }
  tokenIdsToText(tokenIds) {
    const joined = tokenIds.filter((id) => id !== this.blankId).map((id) => this.vocab.get(id) ?? "").filter((token) => token && token !== "<unk>" && token !== "<blank>").join("").replaceAll(WORD_PREFIX, " ");
    return normalizeArabic(joined).trim();
  }
  tokenIdsToRawTokens(tokenIds) {
    return tokenIds.filter((id) => id !== this.blankId).map((id) => this.vocab.get(id) ?? "").filter((token) => token && token !== "<unk>");
  }
  tokenIdsToWordEnds(tokenIds) {
    const tokens = this.tokenIdsToRawTokens(tokenIds);
    const ends = [];
    let inWord = false;
    for (let i = 0; i < tokens.length; i++) {
      const token = tokens[i];
      if (token === WORD_PREFIX || token.startsWith(WORD_PREFIX)) {
        if (inWord) ends.push(i);
        inWord = token !== WORD_PREFIX;
        continue;
      }
      inWord = true;
    }
    if (inWord) ends.push(tokens.length);
    return ends.filter((end, idx) => idx === 0 || end > ends[idx - 1]);
  }
};

// src/quran-text-adapter.ts
function refKey(surah, ayah, ayahEnd = ayah) {
  return `${surah}:${ayah}:${ayahEnd}`;
}
function wordsFromText(text) {
  return text.split(/\s+/).filter(Boolean);
}
function adaptQuranTextData(verses, ctcTokens, decoder) {
  return verses.map((verse) => {
    const clean = normalizeArabic(verse.text_clean ?? verse.text_uthmani);
    const tokenIds = ctcTokens[refKey(verse.surah, verse.ayah)] ?? [];
    const decoded = decoder.tokenIdsToText(tokenIds);
    const joined = decoded || clean;
    const words = wordsFromText(joined);
    return {
      ...verse,
      text_clean: clean,
      phonemes: joined,
      phonemes_joined: joined,
      phoneme_tokens: decoder.tokenIdsToRawTokens(tokenIds),
      phoneme_tokens_no_bsm: null,
      phoneme_token_ids: tokenIds,
      phoneme_token_ids_no_bsm: null,
      word_token_ends: decoder.tokenIdsToWordEnds(tokenIds),
      phonemes_joined_no_bsm: null,
      phonemes_joined_ns: joined.replace(/ /g, ""),
      phonemes_joined_no_bsm_ns: null,
      phoneme_words: words
    };
  });
}
function validateCtcTokenRoundTrip(verses, decoder, sampleSize = 24) {
  const errors = [];
  const step = Math.max(1, Math.floor(verses.length / sampleSize));
  for (let i = 0; i < verses.length; i += step) {
    const verse = verses[i];
    const ids = verse.phoneme_token_ids ?? [];
    if (!ids.length) {
      errors.push(`${verse.surah}:${verse.ayah} has no CTC token ids`);
      continue;
    }
    const decoded = decoder.tokenIdsToText(ids);
    const expected = normalizeArabic(verse.text_clean ?? verse.text_uthmani);
    if (decoded && expected && decoded !== expected) {
      errors.push(`${verse.surah}:${verse.ayah} token decode mismatch`);
    }
  }
  return errors;
}

// src/speech.ts
var SPEECH_DEFAULTS = {
  sampleRate: 16e3,
  frameMs: 30,
  rmsThreshold: 3e-3,
  minVoicedMs: 300
};
function voicedMilliseconds(audio, options = {}) {
  const { sampleRate, frameMs, rmsThreshold } = { ...SPEECH_DEFAULTS, ...options };
  const frame = Math.max(1, Math.round(sampleRate * frameMs / 1e3));
  const thresholdSq = rmsThreshold * rmsThreshold;
  let voicedFrames = 0;
  for (let start = 0; start + frame <= audio.length; start += frame) {
    let sum = 0;
    for (let i = start; i < start + frame; i++) sum += audio[i] * audio[i];
    if (sum / frame > thresholdSq) voicedFrames++;
  }
  return voicedFrames * frameMs;
}
function detectSpeech(audio, options = {}) {
  const { minVoicedMs } = { ...SPEECH_DEFAULTS, ...options };
  return voicedMilliseconds(audio, options) >= minVoicedMs;
}

// src/stitch.ts
var WORD_SIM_THRESHOLD = 0.8;
var MAX_OVERLAP_WORDS = 30;
var MIN_OVERLAP_AGREEMENT = 0.6;
function wordsClose(a, b) {
  return a === b || ratio(a, b) >= WORD_SIM_THRESHOLD;
}
function overlapMerge(prev, next) {
  if (prev.length === 0) return next.slice();
  if (next.length === 0) return prev.slice();
  const maxK = Math.min(prev.length, next.length, MAX_OVERLAP_WORDS);
  let bestK = 0;
  let bestMatched = 0;
  for (let k = 1; k <= maxK; k++) {
    const a = prev.slice(prev.length - k);
    const b = next.slice(0, k);
    let matched = 0;
    for (let i = 0; i < k; i++) if (wordsClose(a[i], b[i])) matched++;
    if (matched >= Math.ceil(k * MIN_OVERLAP_AGREEMENT) && matched >= bestMatched) {
      bestMatched = matched;
      bestK = k;
    }
  }
  return prev.concat(next.slice(bestK));
}
function stitchTranscripts(chunks) {
  const nonEmpty = chunks.map((c) => c.trim()).filter(Boolean);
  if (nonEmpty.length === 0) return "";
  let merged = nonEmpty[0].split(/\s+/);
  for (let i = 1; i < nonEmpty.length; i++) {
    merged = overlapMerge(merged, nonEmpty[i].split(/\s+/));
  }
  return merged.join(" ");
}

// src/judge.ts
var MATCH_SCORE = 2;
var GAP_OPEN = -3;
var GAP_EXTEND = -0.5;
var FUZZY_ALIGNMENT_THRESHOLD = 0.7;
var UNIFY = {
  "\u0622": "\u0627",
  // alef madda -> alef
  "\u0623": "\u0627",
  // alef hamza above -> alef
  "\u0625": "\u0627",
  // alef hamza below -> alef
  "\u0671": "\u0627",
  // alef wasla -> alef
  "\u0624": "\u0648",
  // waw hamza -> waw
  "\u0626": "\u064A",
  // yaa hamza -> yaa
  "\u0649": "\u064A",
  // alef maqsura -> yaa
  "\u0629": "\u0647",
  // taa marbuta -> haa
  "\u0640": ""
  // tatweel removed
};
var PHONETIC = {
  "\u0635": "\u0633",
  // saad -> seen
  "\u0636": "\u062F",
  // daad -> dal
  "\u0637": "\u062A",
  // taa -> ta
  "\u0638": "\u0630",
  // dhaa -> dhal
  "\u0642": "\u0643",
  // qaf -> kaf
  "\u0621": "\u0627",
  // hamza -> alef
  "\u063A": "\u062E",
  // ghain -> khaa
  "\u062B": "\u0633",
  // thaa -> seen
  "\u0630": "\u0632"
  // dhal -> zayn
};
function isDiacriticOrMark(ch) {
  return ch >= "\u0610" && ch <= "\u061A" || ch >= "\u064B" && ch <= "\u065F" || ch === "\u0670" || ch >= "\u06D6" && ch <= "\u06ED";
}
function normalizeArabic2(text) {
  if (!text) return "";
  let out = "";
  for (const ch of text) {
    if (isDiacriticOrMark(ch)) continue;
    const unified = UNIFY[ch];
    const c = unified !== void 0 ? unified : ch;
    for (const cc of c) {
      if (cc === " " || /\s/.test(cc)) out += " ";
      else if (cc >= "\u0600" && cc <= "\u06FF") out += cc;
    }
  }
  return out.split(/\s+/).filter(Boolean).join(" ").trim();
}
function phoneticNormalize(text) {
  const norm = normalizeArabic2(text);
  let out = "";
  for (const ch of norm) out += PHONETIC[ch] ?? ch;
  return out;
}
function lcsLength(a, b) {
  const n = a.length, m = b.length;
  if (!n || !m) return 0;
  let prev = new Array(m + 1).fill(0);
  let cur = new Array(m + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    for (let j = 1; j <= m; j++) {
      cur[j] = a[i - 1] === b[j - 1] ? prev[j - 1] + 1 : Math.max(prev[j], cur[j - 1]);
    }
    [prev, cur] = [cur, prev];
  }
  return prev[m];
}
function ratio2(a, b) {
  if (a === b) return 1;
  const la = a.length, lb = b.length;
  if (!la && !lb) return 1;
  if (!la || !lb) return 0;
  return 2 * lcsLength(a, b) / (la + lb);
}
function partialRatio2(a, b) {
  if (!a.length || !b.length) return 0;
  const [short, long] = a.length <= b.length ? [a, b] : [b, a];
  let best = 0;
  for (let i = 0; i + short.length <= long.length; i++) {
    best = Math.max(best, ratio2(short, long.slice(i, i + short.length)));
    if (best === 1) break;
  }
  return best;
}
var CONFUSION_SETS = [
  ["\u0642", "\u0643"],
  ["\u0635", "\u0633"],
  ["\u0636", "\u062F"],
  ["\u0637", "\u062A"],
  ["\u0638", "\u0630"],
  ["\u0647", "\u0629"],
  ["\u0639", "\u0627"],
  ["\u062D", "\u0647"],
  ["\u063A", "\u062E"],
  ["\u062B", "\u0633"],
  ["\u0630", "\u0632"],
  ["\u0639", "\u0623"]
];
var CONFUSION_MAP = {};
for (const pair of CONFUSION_SETS) {
  for (const c of pair) {
    (CONFUSION_MAP[c] ?? (CONFUSION_MAP[c] = /* @__PURE__ */ new Set())).add(pair[0] === c ? pair[1] : pair[0]);
  }
}
function isConfusion(expNorm, heardNorm) {
  if (expNorm.length !== heardNorm.length || expNorm === heardNorm) return false;
  let diffs = 0, ei = -1;
  for (let i = 0; i < expNorm.length; i++) {
    if (expNorm[i] !== heardNorm[i]) {
      diffs++;
      ei = i;
      if (diffs > 1) return false;
    }
  }
  if (diffs !== 1) return false;
  return CONFUSION_MAP[expNorm[ei]]?.has(heardNorm[ei]) ?? false;
}
function thresholds(asrConfidence) {
  if (asrConfidence == null || asrConfidence >= 0.8) {
    return { correct: 0.9, likely_correct: 0.75, uncertain: 0.6, likely_wrong: 0.45 };
  }
  if (asrConfidence >= 0.5) {
    return { correct: 0.85, likely_correct: 0.7, uncertain: 0.55, likely_wrong: 0.4 };
  }
  return { correct: 0.8, likely_correct: 0.65, uncertain: 0.5, likely_wrong: 0.35 };
}
function matchWord(expected, heard, asrConfidence) {
  const expNorm = normalizeArabic2(expected);
  const heardNorm = normalizeArabic2(heard);
  if (expNorm === heardNorm) return { status: "correct", confidence: 1 };
  if (phoneticNormalize(expected) === phoneticNormalize(heard)) {
    return { status: "likely_correct", confidence: 0.85 };
  }
  const bestFuzzy = Math.max(ratio2(expNorm, heardNorm), partialRatio2(expNorm, heardNorm));
  const confusion = isConfusion(expNorm, heardNorm);
  const th = thresholds(asrConfidence);
  if (bestFuzzy >= th.correct) return { status: "correct", confidence: bestFuzzy };
  if (bestFuzzy >= th.likely_correct || confusion) return { status: "likely_correct", confidence: bestFuzzy };
  if (bestFuzzy >= th.uncertain) return { status: "uncertain", confidence: bestFuzzy };
  if (bestFuzzy >= th.likely_wrong) return { status: "likely_wrong", confidence: bestFuzzy };
  return { status: "wrong", confidence: bestFuzzy };
}
function alignmentScore(exp, heard) {
  if (exp === heard) return MATCH_SCORE;
  const r = ratio2(exp, heard);
  if (r > FUZZY_ALIGNMENT_THRESHOLD) return MATCH_SCORE * r - 1;
  return -1;
}
function alignWords(expected, heard, verseBoundaries = /* @__PURE__ */ new Set()) {
  const m = expected.length, n = heard.length;
  const NEG = Number.NEGATIVE_INFINITY, EPS = 1e-9;
  const mk = () => Array.from({ length: m + 1 }, () => new Array(n + 1).fill(NEG));
  const M = mk(), X = mk(), Y = mk();
  M[0][0] = 0;
  for (let i2 = 1; i2 <= m; i2++) X[i2][0] = GAP_OPEN + (i2 - 1) * GAP_EXTEND;
  for (let j2 = 1; j2 <= n; j2++) Y[0][j2] = GAP_OPEN + (j2 - 1) * GAP_EXTEND;
  for (let i2 = 1; i2 <= m; i2++) {
    const gapOpenI = verseBoundaries.has(i2) ? GAP_OPEN * 0.5 : GAP_OPEN;
    for (let j2 = 1; j2 <= n; j2++) {
      const s = alignmentScore(expected[i2 - 1], heard[j2 - 1]);
      M[i2][j2] = Math.max(M[i2 - 1][j2 - 1], X[i2 - 1][j2 - 1], Y[i2 - 1][j2 - 1]) + s;
      X[i2][j2] = Math.max(M[i2 - 1][j2] + gapOpenI, X[i2 - 1][j2] + GAP_EXTEND);
      Y[i2][j2] = Math.max(M[i2][j2 - 1] + GAP_OPEN, Y[i2][j2 - 1] + GAP_EXTEND);
    }
  }
  const best = Math.max(M[m][n], X[m][n], Y[m][n]);
  let state = Math.abs(best - M[m][n]) < EPS ? "M" : Math.abs(best - X[m][n]) < EPS ? "X" : "Y";
  const out = [];
  let i = m, j = n;
  while (i > 0 || j > 0) {
    if (state === "M" && i > 0 && j > 0) {
      const prevBest = Math.max(M[i - 1][j - 1], X[i - 1][j - 1], Y[i - 1][j - 1]);
      const prev = Math.abs(prevBest - M[i - 1][j - 1]) < EPS ? "M" : Math.abs(prevBest - X[i - 1][j - 1]) < EPS ? "X" : "Y";
      out.push({ ei: i - 1, hi: j - 1, op: expected[i - 1] === heard[j - 1] ? "match" : "substitute" });
      state = prev;
      i--;
      j--;
    } else if (state === "X" && i > 0) {
      const gapOpenI = verseBoundaries.has(i) ? GAP_OPEN * 0.5 : GAP_OPEN;
      const fromX = X[i - 1][j] + GAP_EXTEND;
      state = Math.abs(X[i][j] - fromX) < EPS ? "X" : "M";
      void gapOpenI;
      out.push({ ei: i - 1, hi: null, op: "delete" });
      i--;
    } else if (j > 0) {
      const fromY = Y[i][j - 1] + GAP_EXTEND;
      state = Math.abs(Y[i][j] - fromY) < EPS ? "Y" : "M";
      out.push({ ei: null, hi: j - 1, op: "insert" });
      j--;
    } else break;
  }
  out.reverse();
  return out;
}
function collapse(status) {
  if (status === "correct" || status === "likely_correct") return "correct";
  if (status === "uncertain") return "uncertain";
  return "apparent-error";
}
function judgeAttempt(expectedWords, heardWords, opts = {}) {
  const expNorm = expectedWords.map(normalizeArabic2);
  const heardNorm = heardWords.map(normalizeArabic2);
  const alignment = alignWords(expNorm, heardNorm, opts.verseBoundaries);
  let lastAttemptedEi = -1;
  for (const a of alignment) {
    if ((a.op === "match" || a.op === "substitute") && a.ei != null) {
      lastAttemptedEi = Math.max(lastAttemptedEi, a.ei);
    }
  }
  const words = [];
  let correct = 0, apparentErrors = 0, uncertain = 0, insertions = 0;
  for (const a of alignment) {
    if (a.op === "match") {
      words.push({
        expectedIndex: a.ei,
        recognizedIndex: a.hi,
        expected: expectedWords[a.ei],
        observed: heardWords[a.hi],
        operation: "match",
        judgment: "correct"
      });
      correct++;
    } else if (a.op === "substitute") {
      const conf = opts.confidences?.[a.hi] ?? null;
      const judgment = collapse(matchWord(expectedWords[a.ei], heardWords[a.hi], conf).status);
      words.push({
        expectedIndex: a.ei,
        recognizedIndex: a.hi,
        expected: expectedWords[a.ei],
        observed: heardWords[a.hi],
        operation: "substitution",
        judgment
      });
      if (judgment === "correct") correct++;
      else if (judgment === "uncertain") uncertain++;
      else apparentErrors++;
    } else if (a.op === "delete") {
      if (a.ei <= lastAttemptedEi) {
        words.push({
          expectedIndex: a.ei,
          recognizedIndex: null,
          expected: expectedWords[a.ei],
          operation: "omission",
          judgment: "apparent-error"
        });
        apparentErrors++;
      } else {
        words.push({
          expectedIndex: a.ei,
          recognizedIndex: null,
          expected: expectedWords[a.ei],
          operation: "unattempted",
          judgment: "unattempted"
        });
      }
    } else {
      words.push({
        expectedIndex: null,
        recognizedIndex: a.hi,
        observed: heardWords[a.hi],
        operation: "insertion",
        judgment: "apparent-error"
      });
      insertions++;
    }
  }
  const decided = correct + apparentErrors;
  const attempted = decided + uncertain;
  const coveragePercent = attempted > 0 ? decided / attempted * 100 : 0;
  const scorePercent = attempted > 0 && coveragePercent >= 80 ? 100 * correct / (correct + apparentErrors + insertions) : null;
  const overallDenominator = expectedWords.length + insertions;
  const overallPercent = overallDenominator > 0 ? 100 * correct / overallDenominator : null;
  return { words, attempted, correct, apparentErrors, uncertain, insertions, coveragePercent, scorePercent, overallPercent };
}

// src/index.ts
var CHAMPION_TRUST_THRESHOLD = 0.8;
var FRAGMENT_TRUST_FLOOR = 0.62;
var FRAGMENT_FIT_MIN = 0.75;
var FRAGMENT_FIT_MARGIN_MIN = 0.15;
var FRAGMENT_MIN_CHARS = 12;
var MIN_TRUSTED_LETTERS = 5;
var ARABIC_LETTER_RE = /[\u0621-\u064A\u0671-\u06D3]/g;
function arabicLetterCount(text) {
  return text.match(ARABIC_LETTER_RE)?.length ?? 0;
}
function isTrustedChampion(match, noSpaceLen) {
  if (!match || !match.score) return false;
  if (match.score >= CHAMPION_TRUST_THRESHOLD) return true;
  return match.score >= FRAGMENT_TRUST_FLOOR && (match.frag_fit ?? 0) >= FRAGMENT_FIT_MIN && (match.fit_margin ?? 0) >= FRAGMENT_FIT_MARGIN_MIN && noSpaceLen >= FRAGMENT_MIN_CHARS;
}
function createAsrSession(runner, assets) {
  const decoder = new TextCTCDecoder(assets.vocab, assets.blankId ?? 1024);
  const quranData = adaptQuranTextData(assets.quran, assets.quranCtcTokens, decoder);
  validateCtcTokenRoundTrip(quranData, decoder);
  const db = new QuranDB(quranData, void 0, assets.quranCtcTokens);
  const transcribeRaw = async (audio) => {
    const { logprobs, timeSteps, vocabSize } = await runner.run(audio);
    const greedy = decoder.decode(logprobs, timeSteps, vocabSize);
    const champion = db.bestJoint03Match(greedy.text);
    const noSpaceLen = greedy.text.replace(/ /g, "").length;
    const trustedChampion = arabicLetterCount(greedy.text) >= MIN_TRUSTED_LETTERS && isTrustedChampion(champion, noSpaceLen) ? champion : null;
    const bestGuess = champion ? {
      surah: champion.surah,
      ayah: champion.ayah,
      ayah_end: champion.ayah_end ?? null,
      score: champion.score,
      margin: champion.margin ?? 0,
      frag_fit: champion.frag_fit ?? 0,
      fit_margin: champion.fit_margin ?? 0,
      trusted: !!trustedChampion
    } : void 0;
    return {
      text: greedy.text,
      tokenIds: greedy.tokenIds,
      acoustic: { logprobs, timeSteps, vocabSize, blankId: decoder.getBlankId() },
      championMatch: trustedChampion ?? void 0,
      bestGuess
    };
  };
  return {
    db,
    decoder,
    /** One-shot: transcribe a clip and return the best verse match (0/0 on none). */
    async transcribe(audio) {
      const result = await transcribeRaw(audio);
      const m = result.championMatch;
      if (!m) {
        return {
          surah: 0,
          ayah: 0,
          ayah_end: null,
          score: 0,
          transcript: result.text,
          bestGuess: result.bestGuess
        };
      }
      return {
        surah: m.surah,
        ayah: m.ayah,
        ayah_end: m.ayah_end ?? null,
        score: m.score,
        transcript: result.text,
        bestGuess: result.bestGuess
      };
    },
    /** Low-level one-shot: full result (acoustic log-probs + champion match). */
    transcribeRaw
  };
}
export {
  MIN_TRUSTED_LETTERS,
  QuranDB,
  SPEECH_DEFAULTS,
  TextCTCDecoder,
  adaptQuranTextData,
  alignWords,
  arabicLetterCount,
  createAsrSession,
  detectSpeech,
  distance,
  fragmentScore,
  isTrustedChampion,
  judgeAttempt,
  partialRatio2 as lcsPartialRatio,
  ratio2 as lcsRatio,
  matchWord,
  normalizeArabic,
  normalizeArabic2 as normalizeArabicForJudge,
  overlapMerge,
  partialRatio,
  phoneticNormalize,
  ratio,
  semiGlobalDistance,
  stitchTranscripts,
  validateCtcTokenRoundTrip,
  voicedMilliseconds
};
