from pathlib import Path
p=Path(__file__).resolve().parent/'hash/src/prefix.rs';s=p.read_text().replace('    ram_bytes: usize,','    ram_bytes: usize,\n    bulk_keys: bool,',1)
i=s.index('        let mut out = Self {')
s=s[:i]+s[i:].replace('            ram_bytes: 0,','''            ram_bytes: 0,
            bulk_keys: match std::env::var("DS4_BULK_PREFIX_KEYS").as_deref() {
                Ok("1") => true,
                Ok("0") | Err(std::env::VarError::NotPresent) => false,
                _ => anyhow::bail!("DS4_BULK_PREFIX_KEYS must be 0 or 1"),
            },''',1)
i=s.index('    fn draft_key(')
s=s[:i]+'''    // Checkpoint keys use 32-bit token IDs in little-endian wire order.
    // Feed complete SHA blocks at once without depending on host endianness.
    fn update_tokens(state: &mut Sha256, tokens: &[u32]) {
        let mut bytes = [0u8; 256];
        for group in tokens.chunks(64) {
            for (dst, id) in bytes.chunks_exact_mut(4).zip(group) {
                dst.copy_from_slice(&id.to_le_bytes());
            }
            state.update(&bytes[..4 * group.len()]);
        }
    }

    fn key_at(&self, tokens: &[u32], pos: usize, span: usize) -> Result<Key> {
        if !self.bulk_keys {
            return Ok(self.keys(tokens, pos, span)?.last().context("empty prefix cache key")?.1);
        }
        ensure!(pos > 0 && self.chunk > 0 && pos.is_multiple_of(self.chunk)
                && pos <= tokens.len() && pos <= self.capacity && (1..=128).contains(&span),
                "invalid prefix boundary");
        let mut state = self.identity.clone();
        state.update((span as u32).to_le_bytes());
        Self::update_tokens(&mut state, &tokens[..pos]);
        Ok(state.finalize().into())
    }
'''+s[i:]
s=s.replace('''let base = self
            .keys(tokens, pos, span)?
            .last()
            .context("empty drafter cache key")?
            .1;''','let base = self.key_at(tokens, pos, span)?;')
s=s.replace('''        let key = self
            .keys(tokens, pos, span)?
            .last()
            .context("empty prefix cache key")?
            .1;''','        let key = self.key_at(tokens, pos, span)?;')
s=s.replace('''            for x in chunk {
                state.update(x.to_le_bytes());
            }''','''            if self.bulk_keys {
                Self::update_tokens(&mut state, chunk);
            } else {
                for x in chunk {
                    state.update(x.to_le_bytes());
                }
            }''')
s=s.replace('        for (pos, key) in self.keys(tokens, max_pos, span)?.into_iter().rev() {','''        // Warm agent turns commonly reproduce the last possible boundary.
        // A miss must still search every eligible RAM and disk entry.
        let fast_hit = if self.bulk_keys && max_pos > 0 && max_pos.is_multiple_of(self.chunk) {
            let key = self.key_at(tokens, max_pos, span)?;
            self.ram.iter().any(|(k, _)| *k == key).then_some((max_pos, key))
        } else {
            None
        };
        let candidates = match fast_hit {
            Some(hit) => vec![hit],
            None => self.keys(tokens, max_pos, span)?,
        };
        for (pos, key) in candidates.into_iter().rev() {''')
i=s.index('#[cfg(test)]\nmod tests')
s=s[:i]+s[i:].replace('            ram_bytes: 0,','            ram_bytes: 0,\n            bulk_keys: true,')
i=s.index('    #[test]',i)
s=s[:i]+'''    #[test]
    fn bulk_keys_preserve_existing_target_and_drafter_namespaces() -> Result<()> {
        let ids: Vec<_> = (0..521u32).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
        for chunk in [1, 3, 64, 128] {
            let mut cache = PrefixCache {
                identity: Sha256::new(), draft_identity: [7;32], chunk, capacity: ids.len(),
                ram_limit: 0, disk_limit: 0, dir: None, ram: VecDeque::new(), ram_bytes: 0,
                bulk_keys: false,
            };
            for span in [1, 39, 128] {
                cache.bulk_keys = false;
                let old = cache.keys(&ids, ids.len(), span)?;
                let last = old.last().unwrap().0;
                let draft = cache.draft_key(&ids, last, span)?;
                cache.bulk_keys = true;
                assert_eq!(cache.keys(&ids, ids.len(), span)?, old);
                for (n,key) in old { assert_eq!(cache.key_at(&ids,n,span)?, key); }
                assert_eq!(cache.draft_key(&ids,last,span)?,draft);
            }
        }
        Ok(())
    }
'''+s[i:]
p.write_text(s)
